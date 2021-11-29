{
    # join paths without / repetition
    # for PyTorch/PyTorch Lightning is not strictly required
    joinPath(a, b)::
        if std.endsWith(a, "/") then
            a + b
        else
            a + "/" + b,
}
