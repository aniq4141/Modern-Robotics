import modern_robotics as mr
import numpy as np
from modern_robotics import se3ToVec, MatrixLog6, TransInv, FKinBody, JacobianBody

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    # Initialize joint angles
    thetalist = np.array(thetalist0).copy()
    i = 0  # Iteration counter
    maxiterations = 5  # Maximum number of iterations
    joint_matrix = []  # Initialize matrix to store joint vectors for each iteration

    # Compute the initial error twist in the body frame
    Vb = se3ToVec(MatrixLog6(np.dot(FKinBody(M, Blist, thetalist), TransInv(T))))
    # Compute the initial error check
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev

    # Print initial guess information
    print(f'Iteration {i}:\n\n')
    print('Joint vector :\n', thetalist.T)
    print('SE(3) end?effector config:\n', FKinBody(M, Blist, thetalist)"\n\n")
    print('error twist V_b:\n', Vb"\n")
    print(f'angular error magnitude ||omega_b||: {eomg}\n')
    print(f'linear error magnitude ||v_b||: {ev}\n')

    # Continue iterating until convergence or reaching the maximum number of iterations
    while err and i < maxiterations:
        # Update joint angles using the Jacobian pseudoinverse and the error twist
        thetalist = thetalist + np.dot(np.linalg.pinv(JacobianBody(Blist, thetalist)), Vb)
        i = i + 1  # Increment iteration counter
        # Recalculate the error twist based on the updated joint angles
        Vb = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, thetalist)), T)))
        # Check convergence condition
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev

        # Print out a report for each iteration
        print(f'Iteration {i}:\n\n')
        print('Joint vector :\n', thetalist.T)
        print('SE(3) end?effector config:\n', FKinBody(M, Blist, thetalist)"\n\n")
        print('error twist V_b:\n', Vb"\n")
        print('angular error magnitude ||omega_b||:', np.linalg.norm([Vb[0], Vb[1], Vb[2]]))
        print('linear error magnitude ||v_b||:', np.linalg.norm([Vb[3], Vb[4], Vb[5]]))

        # Save the joint vector for each iteration
        joint_matrix.append(thetalist.tolist())

    # Save the matrix as a .csv file
    if np.linalg.norm([Vb[0], Vb[1], Vb[2]]) <= eomg and np.linalg.norm([Vb[3], Vb[4], Vb[5]]) <= ev:
        print("Desired configuration achieved!")
    else:
        print("Desired configuration not achieved within the specified tolerance.")

    np.savetxt('joint_matrix.csv', joint_matrix, delimiter=',')

    return (thetalist, not err)

# Robot parameters
B = np.array([[0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, -1, 0],
              [0, 1, 1, 1, 0, 1],
              [191, 95, 95, 95, -82, 0],
              [0, -817, -392, 0, 0, 0],
              [817, 0, 0, 0, 0, 0]])

M = np.array([[-1, 0, 0, 817],
              [0, 0, 1, 191],
              [0, 1, 0, -6],
              [0, 0, 0, 1]])

T = np.array([[0, 1, 0, -0.5],
              [0, 0, -1, 0.1],
              [-1, 0, 0, 0.1],
              [0, 0, 0, 1]])

theta = np.array([6.0000, -2.3000, 4.4000, -5.0000, 3.0000, 1.3000])
ew = 0.001  # Angular error tolerance
ev = 0.0001  # Linear error tolerance

# Call the custom inverse kinematics function
result = IKinBodyIterates(B, M, T, theta, ew, ev)
print(result)
