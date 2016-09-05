# Let's us use print() as a function later.
from __future__ import print_function

# Load the file into a list of (H, L, y) tuples
with open("luckeyfxnsample_gVaryHDEHPVaryHwitherror.csv", "r") as handle:
    data = [tuple(float(x) for x in line.split(",")) for line in handle]

def transform(H, L, y,error):
    """Transform (H, L, y) tuples into (y, HL, L, L^2, L^3) tuples."""
    return y, H * L, L, L * L, L * L * L

# Convert data to "linearized" form (where HL, L^2, and L^3 are like more
# variables); by doing so, our function fxn becomes a linear function.
linearized = [transform(*point) for point in data]

#create vectors of raw data to plot
H=[]
L=[]
yexp=[]
e=[]

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def getValueLists(data):
    if len(data)==0:
        test.scatter(L,H,yexp)
        xmin=min(L)-0.1*min(L)
        xmax=max(L)+0.1*min(L)
        ymin=min(H)-0.1*min(H)
        ymax=max(H)+0.1*min(H)
        #test.axis([xmin,xmax,ymin,ymax])
        test.set_ylabel('[H+]')
        test.set_xlabel('[L2-]')
        test.set_zlabel('D0/D-1')
        #plot errorbars
        for i in np.arange(0, len(yexp)):
            test.plot([L[i], L[i]], [H[i], H[i]], [yexp[i]+e[i], yexp[i]-e[i]], marker="_")
    else:
        point=data.pop()
        H.append(point[0])
        L.append(point[1])
        yexp.append(point[2])
        e.append(point[3])
        getValueLists(data)

# Next we use CVXOPT to solve a quadratic program, using this page as a guide:
#       http://cvxopt.org/examples/tutorial/qp.html
# The main documentation for the used function is here:
#       http://cvxopt.org/userguide/coneprog.html#quadratic-programming
# The idea behind this is that we are trying to *minimize* the squared error
# of our estimate \hat{y}, where
#   \hat{y} = B0 * HL + B1 * L + B2 * L^2 + B3 * L^3
# (Apologies for renaming the B's; this is just easier when you have no idea
# what the chemistry or physics behind the original equation is!)
#
# When we compute the squared error (which we wish to minimize, we get):
#   ERR = SUM_i (y_i - \hat{y}_i) ^2
#       = SUM_i (y_i - B0 * HL - B1 * L - B2 * L^2 - B3 * L^3) ^2
# This is an expression quadratic in B0, B1, B2, and B3, which are the values
# we're trying to fit. We wish to *minimize* ERR (our objective or cost
# function). 
# 
# In order to use this, we first have to put our objective into the form:
#       ERR = 1/2 x^T P x + q^T x
# where P is a matrix of our choice, q is a vector of our choice, and x is the 
# combined vector x = [B0, B1, B2, B3]. Before computing this, let's rename our
# coordinates Y, V0, V1, V2, and V3 for consistency (e.g. V0 = HL, V3 = L^3).
# Computing P and q sadly requires us to by hand expand out that ugly squared
# term to get:
#   ERR = SUM_i (-2 b_0 v_0 y-2 b_1 v_1 y-2 b_2 v_2 y-2 b_3 v_3 y+b_0^2 v_0^2
#                +b_1^2 v_1^2+b_2^2 v_2^2+b_3^2 v_3^2+2 b_0 b_1 v_0 v_1
#                +2 b_0 b_2 v_0 v_2+2 b_1 b_2 v_1 v_2+2 b_0 b_3 v_0 v_3
#                +2 b_1 b_3 v_1 v_3+2 b_2 b_3 v_2 v_3+y^2)
# Jesus that's ugly, but hopefuly WA is right. Check it in Mathematica if you
# want! Original source:
#   http://www.wolframalpha.com/input/?i=(y+-+b_0+*+v_0+-+b_1+*+v_1+-+b_2+*+v_2+-+b_3+*+v_3)%5E2
# So let's write a function that, given a single point, computes the
# contribution of it to P and to q:

from cvxopt import matrix, solvers
def Pq_contribution(y, v_0, v_1, v_2, v_3):
    # The linear pieces (contributions to q)
    q = matrix([-2.0 * v_0 * y, # b_0
         -2.0 * v_1 * y, # b_1
         -2.0 * v_2 * y, # b_2
         -2.0 * v_3 * y, # b_3
        ])
    # The quadratic pieces (contributions to P)
    P = matrix([
        [
            2.0*v_0**2, # b_0^2
            2.0 * v_0 * v_1, # b_0 b_1
            2.0 * v_0 * v_2, # b_0 b_2
            2.0 * v_0 * v_3, # b_0 b_3
        ],
        [
            2.0 * v_0 * v_1, # b_0 b_1
            2.0*v_1**2, # b_1^2
            2.0 * v_1 * v_2, # b_1 b_2
            2.0 * v_1 * v_3, # b_1 b_3
        ],
        [
            2.0 * v_0 * v_2, # b_0 b_2
            2.0 * v_1 * v_2, # b_1 b_2
            2.0*v_2 ** 2, # b_2^2
            2.0 * v_2 * v_3, # b_2 b_3
        ],
        [
            2.0 * v_0 * v_3, # b_0 b_3
            2.0 * v_1 * v_3, # b_1 b_3
            2.0 * v_2 * v_3, # b_2 b_3
            2.0*v_3 ** 2, # b_3^2
        ],
    ])

    # We also have a pieces:
    #   y^2
    # But it doesn't matter since it doesn't have any parameters. Constants
    # can't be minimized to just discard this pieces from the objective.

    return P, q

# Compute P and q. Then (visually) check that P is symmetric!
P = None
q = None
for point in linearized:
    if P is None:
        P, q = Pq_contribution(*point)
    else:
        P_contrib, q_contrib = Pq_contribution(*point)
        P += P_contrib
        q += q_contrib

print("P:", P)
print("q:", q)

# Well, these numbers are vastly different magnitude and order, so this may not
# actually work. But we can try?

# Next up, we need to set up the constraints. Constraints are of the form:
#       G x <= h
# These are a bit easier to set up. We want:
#   B0 > B1
#   B3 > B2
#   B2 > B1
#   B1 > 0
# Equivalently:
#   0 > B1 - B0
#   0 > B2 - B3
#   0 > B1 - B2
#   0 > - B1
# Thus, we can let h = 0 (vector of zeros), and let G be the following matrix:
G = matrix([
    [-1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 1.0, -1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
]).ctrans()
h = matrix([0.0, 0.0, 0.0, 0.0])

print("G:", G)
print("h:", h)

# Finally, we have everything set up to solve! Let's try it... hesitantly...
# eager with anticipation... oh no... what will happen...
solution = solvers.qp(P, q, G, h)

# Print solution and helpful message.

print()
print("Optimal solution? " + "Yes!" if solution["status"] == "optimal" else "No :(")
print("Solution:", solution['x'])
print("Does this make *any* physical sense? I don't know. That's up to you!")

# Plot Raw and fitted data together
import numpy as np

def FitPlotter(param):
    xH=np.linspace(min(H),max(H),3*len(H))
    xL=np.linspace(min(L),max(L),3*len(L))
    y=param[0]*xH*xL+param[1]*xL+param[2]*xL**2+param[3]*xL**3
    test.plot_wireframe(xL,xH,y)

Beta0, Beta1, Beta2, Beta3 = solution['x'][0], solution['x'][1], solution['x'][2], solution['x'][3]

fig=plt.figure()
test=fig.add_subplot(111,projection='3d')
getValueLists(data)
FitPlotter([Beta0, Beta1, Beta2, Beta3])
plt.show()


print("Hopefully at least this demo helps you in some way, though!")
