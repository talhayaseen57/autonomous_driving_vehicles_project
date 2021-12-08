from colour import Color


def generate_gradient(c1, c2, n=10):
    """takes two colors as arguments
    and an optional argument for the number of samples between the two colors
    and returns a list of colors represnting a gradient"""

    color1 = Color(c1)
    color2 = Color(c2)
    gradient = list(color1.range_to(color2, n))
    return gradient


def get_n_colors(n):
    """takes an integer n as an argument
    and returns a list of colors with length n"""

    colors = []
    for i in range(n):
        colors.append(Color(pick_for=str(i)))
