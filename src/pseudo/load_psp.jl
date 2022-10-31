import PeriodicTable

"""Return the data directory with pseudopotential files"""
datadir_psp() = joinpath(get(ENV, "DFTK_DATADIR", DFTK_DATADIR), "psp")


"""
Load a pseudopotential file from the library of pseudopotentials.
The file is searched in the directory `datadir_psp()` and by the `key`.
If the `key` is a path to a valid file, the extension is used to determine
the type of the pseudopotential file format and a respective class is returned.
"""
function load_psp(key::AbstractString)
    if startswith(key, "hgh/") || endswith(lowercase(key), ".hgh")
        pseudo_type = PspHgh
        extension = ".hgh"
    elseif endswith(lowercase(key), ".upf")
        pseudo_type = PspUpf
        extension = ".upf"
    else
        error("Could not determine pseudopotential family of '$key'")
    end

    Sys.iswindows() && (key = replace(key, "/" => "\\"))
    if isfile(key)  # Key is a file ... deduce identifier
        fullpath = key
        identifier = replace(key, "\\" => "/")
        if startswith(identifier, datadir_psp())
            identifier = identifier[length(datadir_psp())+1:end]
        end
    elseif startswith(lowercase(key), r"http://|https://")  # Key is a URL .. download it
        fullpath = Downloads.download(key, joinpath(tempdir(), "psp$(extension)"))
        identifier = key
    else
        # Not a file: Treat as identifier, add extension if needed
        fullpath = joinpath(datadir_psp(), lowercase(key))
        isfile(fullpath) || (fullpath = fullpath * extension)
        identifier = replace(lowercase(key), "\\" => "/")
    end

    if isfile(fullpath)
        return pseudo_type(fullpath; identifier)
    else
        error("Could not find pseudopotential for identifier " *
              "'$identifier' in directory '$(datadir_psp())'")
    end
end
