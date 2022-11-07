include("run_scf_and_compare.jl")
include("testcases.jl")

function run_iron_pbe(T; kwargs...)
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 20. They are not yet converged and thus require the same discretisation
    # parameters to be obtained.
    ref_evals = [
        [0.0603597727989307, 0.1964963273638626, 0.196496327424440, 0.279192222553112,
         0.2791922225741613, 0.3415221335998876, 0.837882559419754, 0.883850560591423,
         0.8838505606211768, 1.3135367355436536],
        [0.1384929268069029, 0.1847168453364975, 0.223179759800174, 0.320070899985990,
         0.3500724891746176, 0.4685757607370267, 0.541752194212558, 0.751365680734661,
         0.8039132927796911, 1.3939297677405071],
        [-0.017996603976028, 0.2383855826934185, 0.238385582734711, 0.248204676138927,
         0.2509395500598295, 0.2776437400588896, 1.069915401940919, 1.088217176897224,
         1.094997859335961,  1.0949978593466851],
        [0.1102557166995405, 0.2077201723056727, 0.220685303120809, 0.289884460857327,
         0.3490062808992303, 0.3571047250832524, 0.664551132243957, 0.890354172420178,
         0.939822681382406,  1.2259972985258636],
        [0.1723514110126840, 0.1723514110181127, 0.189598224957126, 0.315084007273243,
         0.3150840073174671, 0.5487559496577702, 0.548755949657792, 0.571153866844390,
         1.0611134432316718, 1.1887518709297569],
        [0.1360541296075938, 0.1413608406233668, 0.337616953214017, 0.337616953257584,
         0.3463728840905585, 0.4304010493995122, 0.688627292839765, 0.688627292852315,
         0.885008380770321,  0.9722786718518246],
        [0.0802990962833626, 0.3488798033726516, 0.348879803416372, 0.533263624117060,
         0.560354114948579,  0.5603541149670136, 0.923281827089562, 0.967838872125574,
         0.9678388721641925, 1.300215418446228],
        [0.2341496631160049, 0.2737567834221212, 0.320646675118266, 0.590600827614029,
         0.6440928824646408, 0.6458637753212415, 0.678343515679297, 0.838647690182280,
         0.8763210347583158, 1.4092936521531203],
        [-0.002234753604747, 0.4096246186291687, 0.409624618662776, 0.434260327970128,
         0.5068101375084778, 0.5757957165012942, 1.137207834311533, 1.137826252874365,
         1.170363096833071,  1.170363096849632],
        [0.1518900787487526, 0.3293780680641614, 0.376401550325491, 0.512562269331525,
         0.5557310122303195, 0.6261449425921871, 0.794097184155989, 0.967295197092196,
         1.0000550921659532, 1.2999173820510477],
        [0.2873355363445261, 0.2873355363447599, 0.319313192152575, 0.537629072823137,
         0.5376290728591641, 0.6802062250711767, 0.704199805731151, 0.704199805731498,
         1.1322730987840155, 1.255912074880981],
        [0.2512356397409882, 0.315293666807424,  0.491297439253523, 0.4912974392811193,
         0.5558649368408816, 0.556692128645629,  0.777563890322163, 0.7775638903489546,
         0.9998569230219644, 1.1313796020728688],
    ]
    ref_etot = -18.21465922614397
    ref_magn = 2.98199463

    # Produce reference data and guess for this configuration
    Fe = ElementPsp(iron_bcc.atnum, psp=load_psp("hgh/lda/Fe-q8.hgh"))
    atoms, positions = [Fe], [zeros(3)]
    magnetic_moments = [4.0]
    model = model_PBE(Array{T}(iron_bcc.lattice), iron_bcc.atoms, iron_bcc.positions;
                      temperature=0.01, magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut=20, fft_size=[20, 20, 20],
                           kgrid=[4, 4, 4], kshift=[1/2, 1/2, 1/2])

    scfres = run_scf_and_compare(T, basis, ref_evals, ref_etot;
                                 ρ=guess_density(basis, AutoGuessDensity(), magnetic_moments),
                                 kwargs...)

    magnetisation = sum(spin_density(scfres.ρ)) * basis.dvol
    @test magnetisation ≈ ref_magn atol=5e-5
end


if !isdefined(Main, :FAST_TESTS) || !FAST_TESTS
    @testset "Iron PBE (Float64)" begin
        run_iron_pbe(Float64, test_tol=5e-6, scf_tol=1e-12)
    end
end
