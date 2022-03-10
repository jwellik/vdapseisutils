# Basic utilities

def varinfo(var):

    vname = 'varname'
    vtype = type(var)
    if vtype == 'list':
        vshape = shape(var)
    if vtype == 'ndarray':
        vshape = var.shape()
    else:
        vshape = '?'

    print(':' * 30)
    print('{:10} ({:10}) ({:10})'.format(vname,vtype,vshape))
    print(var)
    print(':'*30)