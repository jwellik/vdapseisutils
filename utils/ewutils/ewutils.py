def lay(depth, Vp_top):
    """Prints Earthworm layer commands"""

    for d, v in zip(depth, Vp_top):
        print("lay {:>6.4f}  {:>6.4f}".format(d, v))
    print()


def velocityd(depth=[0.0, 4.0, 9.0, 16.0, 20.0, 25.0, 41.0], velocity=[5.40, 6.38, 6.59, 6.73, 6.86, 6.95, 7.80],
              header="# Generic Velocity Model",
              ):

    template = "{layer_lines}"  # There is no header to this file

    """
    lay   0.0  5.40
    lay   4.0  6.38
    lay   9.0  6.59
    lay  16.0  6.73
    lay  20.0  6.86
    lay  25.0  6.95
    lay  41.0  7.80 
    """

    layer_line = "lay  {:>4.1f} {:>4.2f}\n"
    layer_lines = "#"+header+"\n"
    for d, v in zip(depth, velocity):

        layer_line.format(d, v)
        layer_lines += layer_line.format(d, v)

    # Fill template
    # name = filename if name is None else name
    template = template.format(layer_lines=layer_lines)

    # Print and Save
    print(template)
    print()

    return template


def velocitycrh(depth=[0.0, 4.0, 9.0, 16.0, 20.0, 25.0, 41.0], velocity=[5.40, 6.38, 6.59, 6.73, 6.86, 6.95, 7.80],
              header="# Generic Velocity Model",
              ):
    template = "{layer_lines}"  # There is no header to this file

    """
    R0 Rabaul Generic
     1.7  0.0
     2.2  0.7
     3.5  1.0
     4.0  2.0
     4.5  3.0
     6.2  5.0
     6.4 15.0
    """

    layer_line = " {:>4.2f} {:>4.1f}\n"
    layer_lines = ""
    layer_lines += header + "\n"
    for d, v in zip(velocity, depth):

        layer_line.format(d, v)
        layer_lines += layer_line.format(d, v)

    # Fill template
    # name = filename if name is None else name
    template = template.format(layer_lines=layer_lines)

    print(template)
    print()

    return template