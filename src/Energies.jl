"""
A simple struct to contain a vector of energies, and utilities to print them in a nice format.
"""
struct Energies{T <: Number}
    # energies["TermName"]
    # parametrization on T acts as a nice check that all terms return correct type
    energies::Vector{Pair{String, T}}
end

function Base.show(io::IO, energies::Energies)
    print(io, "Energies(total = $(energies.total))")
end
function Base.show(io::IO, ::MIME"text/plain", energies::Energies)
    println(io, "Energy breakdown (in Ha):")
    for (name, value) in energies.energies
        @printf io "    %-20s%-10.7f\n" string(name) value
    end
    @printf io "\n    %-20s%-15.12f" "total" energies.total
end
Base.getindex(energies::Energies, i) = only([val for (key, val) in energies.energies if key == i])
Base.values(energies::Energies)      = [val      for (key, val) in energies.energies]
Base.keys(energies::Energies)        = [key      for (key, val) in energies.energies]
Base.pairs(energies::Energies)       = [key=>val for (key, val) in energies.energies]
Base.iterate(energies::Energies)     = iterate(energies.energies)
Base.iterate(energies::Energies, state) = iterate(energies.energies, state)
Base.haskey(energies::Energies, key) = any(==(key), keys(energies))


function Energies(term_types::Vector, energies::Vector{T}) where {T}
    # nameof is there to get rid of parametric types
    Energies{T}([string(nameof(typeof(term))) => energies[i] 
                 for (i, term) in enumerate(term_types)])
end

function Base.propertynames(energies::Energies, private::Bool=false)
    ret = keys(energies)
    append!(ret, "total")
    private && append!(ret, "energies")
end
function Base.getproperty(energies::Energies, x::Symbol)
    x == :total && return sum(values(energies))
    x == :energies && return getfield(energies, x)
    energies.energies[string(x)]
end
