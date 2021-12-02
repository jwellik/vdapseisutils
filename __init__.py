# Basic utilities

def varinfo(var):

    vname = 'varname'
    vtype = type(var)
    if vtype is 'list':
        vshape = shape(var)
    if vtype is 'ndarray':
        vshape = var.shape()
    else:
        vshape = '?'

    print(':' * 30)
    print('{:10} ({:10}) ({:10})'.format(vname,vtype,vshape))
    print(var)
    print(':'*30)