{
    # join paths without / repetition
    # for PyTorch/PyTorch Lightning is not strictly required
    joinPath(a, b)::
        if std.endsWith(a, "/") then
            a + b
        else
            a + "/" + b
    ,

    safeGet(o, f, default=null, include_hidden=true)::
        if include_hidden then
            if std.objectHasAll(o, f) then
                o[f]
            else
                default
        else
            if std.objectHas(o, f) then
                o[f]
            else
                default
    ,
}
