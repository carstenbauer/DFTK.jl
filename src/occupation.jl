import Roots

# Functions for finding the Fermi level and occupation numbers for bands
# The goal is to find εF so that
# f_i = filled_occ * f((εi-εF)/θ) with θ = temperature
# sum_i f_i = n_electrons
# If temperature is zero, (εi-εF)/θ = ±∞.
# The occupation function is required to give 1 and 0 respectively in these cases.
#
# For finite temperature we right now just use bisection; note that with MP smearing
# there might be multiple possible Fermi levels. This could be sped up
# with more advanced methods (e.g. false position), but more care has to
# be taken with convergence criteria and the like

abstract type FermiLevelAlgorithm end
struct ZeroTemperature <: FermiLevelAlgorithm end
struct Bisection       <: FermiLevelAlgorithm end
struct GaussianNewton  <: FermiLevelAlgorithm end

function excess_n_electrons(basis::PlaneWaveBasis, eigenvalues, εF; smearing, temperature)
    occupation = compute_occupation(basis, eigenvalues, εF;
                                    smearing, temperature).occupation
    weighted_ksum(basis, sum.(occupation)) - basis.model.n_electrons
end

function check_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, εF; smearing, temperature) where {T}
    # Sanity check on Fermi level
    excess = excess_n_electrons(basis, eigenvalues, εF; smearing, temperature)
    dexcess = ForwardDiff.derivative(εF) do εF
        excess_n_electrons(basis, eigenvalues, εF; smearing, temperature)
    end
    if abs(excess / basis.model.n_electrons) > sqrt(eps(T))
        if iszero(temperature)
            error("Unable to find non-fractional occupations that have the " *
                  "correct number of electrons. You should add a temperature.")
        else
            error("This should not happen ... debug me.")
        end
    end
    dexcess < -sqrt(eps(T)) && @warn(
        "Negative electron versus Fermi level derivative encountered. Expect an unphysical " *
        "(negative) density of states at εF. Try increasing the number of k-Points or " *
        "switching to a different smearing function."
    )
end

"""
Find the occupation and Fermi level.
"""
function compute_occupation(basis::PlaneWaveBasis{T}, eigenvalues,
                            algorithm::FermiLevelAlgorithm=FermiExtrapolation();
                            temperature=basis.model.temperature,
                            smearing=basis.model.smearing) where {T}
    if !isnothing(basis.model.εF)  # fixed Fermi level
        εF = basis.model.εF
    else  # fixed n_electrons
        # Check there are enough bands
        max_n_electrons = (filled_occupation(basis.model)
                           * weighted_ksum(basis, length.(eigenvalues)))
        if max_n_electrons < basis.model.n_electrons - sqrt(eps(T))
            error("Could not obtain required number of electrons by filling every state. " *
                  "Increase n_bands.")
        end

        @timing "compute_fermi_level" begin
            εF = compute_fermi_level(basis, eigenvalues, algorithm; temperature, smearing)
        end
        check_fermi_level(basis, eigenvalues, εF; smearing, temperature)
    end
    compute_occupation(basis, eigenvalues, εF; temperature, smearing)
end

"""Compute the occupations, given eigenenergies and a Fermi level"""
function compute_occupation(basis::PlaneWaveBasis{T}, eigenvalues, εF::Number;
                            temperature=basis.model.temperature,
                            smearing=basis.model.smearing) where {T}
    # This is needed to get the right behaviour for special floating-point types
    # such as intervals.
    inverse_temperature = iszero(temperature) ? T(Inf) : 1/temperature

    filled_occ = filled_occupation(basis.model)
    occupation = map(eigenvalues) do εk
        occ = filled_occ * Smearing.occupation.(smearing, (εk .- εF) .* inverse_temperature)
        to_device(basis.architecture, occ)
    end
    (; occupation, εF)
end


function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, ::ZeroTemperature;
                             temperature=nothing, smearing=nothing) where {T}
    filled_occ  = filled_occupation(basis.model)
    n_electrons = basis.model.n_electrons
    n_spin = basis.model.n_spin_components

    # Sanity check that we can indeed fill the appropriate number of states
    if n_electrons % (n_spin * filled_occ) != 0
        error("$n_electrons electrons cannot be attained by filling states with " *
              "occupation $filled_occ. Typically this indicates that you need to put " *
              "a temperature or switch to a calculation with collinear spin polarization.")
    end
    n_fill = div(n_electrons, n_spin * filled_occ, RoundUp)

    # For zero temperature, two cases arise: either there are as many bands
    # as electrons, in which case we set εF to the highest energy level
    # reached, or there are unoccupied conduction bands and we take
    # εF as the midpoint between valence and conduction bands.
    if n_fill == length(eigenvalues[1])
        εF = maximum(maximum, eigenvalues) + 1
        εF = mpi_max(εF, basis.comm_kpts)
    else
        # highest occupied energy level
        HOMO = maximum([εk[n_fill] for εk in eigenvalues])
        HOMO = mpi_max(HOMO, basis.comm_kpts)
        # lowest unoccupied energy level, be careful that not all k-points
        # might have at least n_fill+1 energy levels so we have to take care
        # of that by specifying init to minimum
        LUMO = minimum(minimum.([εk[n_fill+1:end] for εk in eigenvalues]; init=T(Inf)))
        LUMO = mpi_min(LUMO, basis.comm_kpts)
        εF = (HOMO + LUMO) / 2
        end
end

function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, ::Bisection;
                             temperature, smearing) where {T}
    if iszero(temperature)
        return compute_fermi_level(basis, eigenvalues, ZeroTemperature())
    end

    # Get rough bounds to bracket εF
    min_ε = minimum(minimum, eigenvalues) - 1
    min_ε = mpi_min(min_ε, basis.comm_kpts)
    max_ε = maximum(maximum, eigenvalues) + 1
    max_ε = mpi_max(max_ε, basis.comm_kpts)

    excess(εF) = excess_n_electrons(basis, eigenvalues, εF; smearing, temperature)
    @assert excess(min_ε) < 0 < excess(max_ε)
    Roots.find_zero(excess, (min_ε, max_ε), Roots.Bisection(), atol=eps(T))
end

function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, ::GaussianNewton;
                             temperature, smearing) where {T}
    # Compute a guess using a monotonic smearing
    smearing_guess = Smearing.Gaussian()
    if smearing isa Smearing.FermiDirac
        smearing_guess = smearing
    end
    εF_guess = compute_fermi_level(basis, eigenvalues, Bisection();
                                   temperature, smearing=smearing_guess)

    excess(εF)  = excess_n_electrons(basis, eigenvalues, εF; smearing, temperature)
    dexcess(εF) = ForwardDiff.derivative(excess, εF)
    if abs(excess(εF_guess) / basis.model.n_electrons) < sqrt(eps(T))
        return εF_guess  # Early exit
    end

    # Really weird ... why use quadratic here ...
    εF = εF_guess
    objective(εF)  = abs2(excess(εF))
    dobjective(εF) = ForwardDiff.derivative(objective, εF)
    try
        εF = Roots.find_zero((objective, dobjective), εF_guess, Roots.Newton(), atol=eps(T))
    catch e
        (e isa Roots.ConvergenceFailed) || rethrow()
    end
    if abs(excess(εF) / basis.model.n_electrons) > sqrt(eps(T))
        # If Newton has issues, fall back to bisection ...
        @warn "Newton algorithm failed to determine Fermi level. Falling back to Bisection."
        return compute_fermi_level(basis, eigenvalues, Bisection(); temperature, smearing)
    else
        return εF
    end
end

# FermiExtrapolation works by extrapolating the Fermi level found using a monotonic smearing
# (f₀) to the Fermi level of the desired smearing function f₁.
#     g₀(εF) = ∑_i f₀((ε-εF) / T) - N
#     g₁(εF) = ∑_i f₁((ε-εF) / T) - N
#
#     g(εF, α) = (1-α) g₀(εF) + g₁(εF)
#
# The constraint g(εF, α) = 0 defines an implicit function εF(α). By differentiating the
# constraint twice we find εF' and εF''.
#
# 0 = dg/dα = (∂g/∂εF) (∂εF/∂α) + (∂g/∂α) (∂α/∂α)
# thus
#     grad = (∂εF/∂α) = - (∂g/∂εF)^{-1} * (∂g/∂α)
#
# 0 = d²g/d²α =   (∂/∂εF) [(∂g/∂εF) (∂εF/∂α) + (∂g/∂α)] (∂εF/∂α)
#               + (∂/∂α)  [(∂g/∂εF) (∂εF/∂α) + (∂g/∂α)] (∂α/∂α)
#             =   [(∂²g/∂²εF)  (∂εF/∂α) + (∂²g/∂α∂εF)] (∂εF/∂α)
#               + [(∂²g/∂α∂εF) (∂εF/∂α) + (∂g/∂εF) (∂²εF/∂²α) + (∂²g/∂α²)]
#             = [(∂²g/∂²εF) (∂εF/∂α) + 2 (∂²g/∂α∂εF)] (∂εF/∂α) + (∂g/∂εF) (∂²εF/∂²α)
# since (∂²g/∂α²) = 0
# thus
#     hess = (∂²εF/∂²α) = - (∂g/∂εF)^{-1} (∂εF/∂α) [(∂²g/∂²εF) (∂εF/∂α) + 2 (∂²g/∂α∂εF)]
# which makes up a model
#     model(α) = εF₀ + δεF(α) = εF₀ + grad * (α - α₀) + 1/2 hess (α - α₀)^2
#
# To ensure the validity of the second-order model, we check that g(εF(α), α) does not get
# too large.
#
Base.@kwdef struct FermiExtrapolation <: FermiLevelAlgorithm
    maxiter = 10
    occupation_tolerance = 1e-4
    verbose = false
end
function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, method::FermiExtrapolation;
                             temperature, smearing) where {T}
    # Early exit ...
    if iszero(temperature)
        return compute_fermi_level(basis, eigenvalues, ZeroTemperature(); temperature, smearing)
    elseif smearing isa Smearing.FermiDirac || smearing isa Smearing.Gaussian
        return compute_fermi_level(basis, eigenvalues, Bisection(); temperature, smearing)
    end

    # The excess of electrons function and its derivatives
    function g(α, εF)
        ((1-α) * excess_n_electrons(basis, eigenvalues, εF;
                                     smearing=Smearing.Gaussian(), temperature)
         + α * excess_n_electrons(basis, eigenvalues, εF; smearing, temperature))
    end
    g_ε(α, εF)  = ForwardDiff.derivative(εF ->   g(α, εF), εF)
    g_εε(α, εF) = ForwardDiff.derivative(εF -> g_ε(α, εF), εF)
    g_α(α, εF)  = ForwardDiff.derivative(α  ->   g(α, εF), α)
    g_αε(α, εF) = ForwardDiff.derivative(α  -> g_ε(α, εF), α)

    # Compute a guess using a monotonic smearing
    εF = compute_fermi_level(basis, eigenvalues, Bisection();
                             temperature, smearing=Smearing.Gaussian())

    if abs(g(1.0, εF) / basis.model.n_electrons) < sqrt(eps(T))
        return εF  # Early exit
    end

    αF = 0.0
    for i in 1:method.maxiter
        if method.verbose && mpi_master()
            println("")
            println("-------  Iter $i -- αF=$αF εF=$εF --------")
            println("")
        end

        # Construct model for εF(α)
        grad = - g_α(αF, εF) / g_ε(αF, εF)
        hess = - grad * (g_εε(αF, εF) * grad + 2g_αε(αF, εF)) / g_ε(αF, εF)
        model_εF     = α -> εF + grad * (α-αF) + hess * (α-αF)^2 / 2

        # Find a range for which the model is valid
        α_trials = [α for α in range(0.1, 1.0, length=method.maxiter) if α > αF]
        αF = α_trials[1]
        for α_trial in α_trials
            error = abs(g(α_trial, model_εF(α_trial)))
            if error < method.occupation_tolerance
                αF = α_trial
            else
                break
            end
        end
        @assert 0.0 < αF ≤ 1.0
        εF = model_εF(αF)

        if isone(αF)
            break
        else
            εF = Roots.find_zero(εF -> g(αF, εF), εF, Roots.Order0();
                                 atol=method.occupation_tolerance / 10, method.verbose)
        end
    end

    if method.verbose && mpi_master()
        println("")
        println("-------  Finish  εF=$εF --------")
        println("")
    end
    Roots.find_zero(εF -> g(αF, εF), εF, Roots.Order0(); atol=eps(T), method.verbose)
end
