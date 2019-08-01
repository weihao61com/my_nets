import numpy as np
import math


class RotationSequence:
    """Enums for the sequence of rotation matrix multiplication.
        e.g. XYZ means the rotation matrix would rotate the object
        along with Z first and Y and X"""
    XYZ, XZY, YXZ, YZX, ZXY, ZYX = range(0, 6)

class BlueNoteSensorRotation:
    """Rotation class to handle blue note sensors. Bluenote sensor frame is defined as
        +X: moving direction
        +Y: left
        +Z: up
        All angles along the axes increase counter clockwise."""
    _degree_to_rad = math.pi/180.0
    _rad_to_degree = 180.0/math.pi

    @staticmethod
    def rotation_matrix_x(angle):
        """Rotation along with x axis"""
        angle_rad = angle * BlueNoteSensorRotation._degree_to_rad
        cos = np.cos(angle_rad)
        sin = np.sin(angle_rad)
        if np.isscalar(cos):
            unity = 1.0
            empty = 0.0
        else:
            unity = np.array([1.0] * len(cos))
            empty = np.array([0.0] * len(cos))

        m = np.array([[unity, empty, empty],
                     [empty, cos, -sin],
                     [empty, sin, cos]])
        return m

    @staticmethod
    def rotation_matrix_y(angle):
        """Rotation along with y axis"""
        angle_rad = angle * BlueNoteSensorRotation._degree_to_rad
        cos = np.cos(angle_rad)
        sin = np.sin(angle_rad)

        if np.isscalar(cos):
            unity = 1.0
            empty = 0.0
        else:
            unity = np.array([1.0] * len(cos))
            empty = np.array([0.0] * len(cos))
        m = np.array([[cos, empty, sin],
                     [empty, unity, empty],
                     [-sin, empty, cos]])
        return m

    @staticmethod
    def rotation_matrix_z(angle):
        """Rotation along with z axis"""
        angle_rad = angle * BlueNoteSensorRotation._degree_to_rad
        cos = np.cos(angle_rad)
        sin = np.sin(angle_rad)

        if np.isscalar(cos):
            unity = 1.0
            empty = 0.0
        else:
            unity = np.array([1.0] * len(cos))
            empty = np.array([0.0] * len(cos))
        m = np.array([[cos, -sin, empty],
                     [sin, cos, empty],
                     [empty, empty, unity]])
        return m

    @staticmethod
    def matrix_multiplication(matrix_a, matrix_b):
        """vectorized version of matrix multiplication
           For matrix vector multiplication you need to make sure the vector has as many entries as the matrix
           and that there is the same number of dimensions"""

        mat_a_shape = matrix_a.shape
        mat_b_shape = matrix_b.shape
        if mat_a_shape[0] != mat_b_shape[0] or mat_a_shape[-1] != mat_b_shape[-1]:
            raise TypeError("Input data is not consistent shape for multiplication")
        if len(matrix_a.shape) == 2:
            return matrix_a.dot(matrix_b)
        # if len(matrix_a.shape) != len(matrix_b.shape):
        dim = len(matrix_a[0, 0, :])

        matrix_a = np.transpose(matrix_a, (2, 1, 0)).reshape(dim, mat_a_shape[0], mat_a_shape[1], 1)
        if len (mat_b_shape) == 3:
            matrix_b = np.transpose(matrix_b, (2, 0, 1)).reshape(dim, mat_b_shape[0], 1, mat_b_shape[1])
            return np.transpose(np.sum(matrix_a * matrix_b, -3), (1, 2, 0))
        elif len (mat_b_shape) == 2:
            matrix_b = np.transpose(matrix_b, (1, 0)).reshape(mat_b_shape[1], mat_b_shape[0])
            return np.transpose(np.sum(matrix_a * matrix_b, -2), (1, 0))
        else:
            raise Exception("Input dimensions of second matrix do not match matrix-matrix or matrix-vector multiplication rules.")


    @staticmethod
    def rotation_matrix(angle_x, angle_y, angle_z, sequence=RotationSequence.ZYX):
        """Computes rotation matrix given rotation angles on x, y and z axes.
        :param angle_x: rotation along x axis
        :param angle_y: rotation along y axis
        :param angle_z: rotation along z axis
        :param sequence: rotation sequence, i.e. ZYX means rotating w.r.t X axis first, Y axis and Z axis.
        :return 3x3 numpy array"""

        rot_x = BlueNoteSensorRotation.rotation_matrix_x(angle_x)
        rot_y = BlueNoteSensorRotation.rotation_matrix_y(angle_y)
        rot_z = BlueNoteSensorRotation.rotation_matrix_z(angle_z)

        if sequence == RotationSequence.XYZ:
            return BlueNoteSensorRotation.matrix_multiplication(rot_x,
                                                         BlueNoteSensorRotation.matrix_multiplication(rot_y,
                                                                                                      rot_z))
        elif sequence == RotationSequence.XZY:
            return BlueNoteSensorRotation.matrix_multiplication(rot_x,
                                                         BlueNoteSensorRotation.matrix_multiplication(rot_z,
                                                                                                      rot_y))
        elif sequence == RotationSequence.YXZ:
            return BlueNoteSensorRotation.matrix_multiplication(rot_y,
                                                         BlueNoteSensorRotation.matrix_multiplication(rot_x,
                                                                                                      rot_z))
        elif sequence == RotationSequence.YZX:
            return BlueNoteSensorRotation.matrix_multiplication(rot_y,
                                                         BlueNoteSensorRotation.matrix_multiplication(rot_z,
                                                                                                      rot_x))
        elif sequence == RotationSequence.ZXY:
            return BlueNoteSensorRotation.matrix_multiplication(rot_z,
                                                         BlueNoteSensorRotation.matrix_multiplication(rot_x,
                                                                                                      rot_y))
        elif sequence == RotationSequence.ZYX:
            return BlueNoteSensorRotation.matrix_multiplication(rot_z,
                                                         BlueNoteSensorRotation.matrix_multiplication(rot_y,
                                                                                                      rot_x))

    @staticmethod
    def rotation_matrix_zxy(angle_x, angle_y, angle_z):
        """Computes rotation matrix given rotation angles on x, y and z axes.
        The rotation matrix is formed by rotating w.r.t Y axis first, X and Z.
        :param angle_x: rotation along x axis
        :param angle_y: rotation along y axis
        :param angle_z: rotation along z axis
        :return 3x3 numpy array"""
        angle_x_rad = angle_x * BlueNoteSensorRotation._degree_to_rad
        angle_y_rad = angle_y * BlueNoteSensorRotation._degree_to_rad
        angle_z_rad = angle_z * BlueNoteSensorRotation._degree_to_rad

        cx = np.cos(angle_x_rad)
        sx = np.sin(angle_x_rad)

        cy = np.cos(angle_y_rad)
        sy = np.sin(angle_y_rad)

        cz = np.cos(angle_z_rad)
        sz = np.sin(angle_z_rad)

        return np.array([[cz*cy-sz*sx*sy, -sz*cx, cz*sy+sz*sx*cy],
                     [sz*cy+cz*sx*sy, cz*cx, sz*sy-cz*sx*cy],
                     [-cx*sy, sx, cx*cy]])


    @staticmethod
    def get_rotation_angles(rot, sequence = RotationSequence.ZYX):
        """Computes rotation angles on x, y and z axes based on the rotation sequence.
        :param rot: rotation matrix. numpy 3x3 array
        :param sequence: rotation sequence, i.e. ZYX means rotating w.r.t X axis first, Y axis and Z axis.
        :return rotation angle x, rotation angle y, rotation angle z"""
        rot_x = None
        rot_y = None
        rot_z = None
        if sequence == RotationSequence.XYZ:
            rot_x = math.atan2(-rot[1][2], rot[2][2])
            rot_y = math.atan2(rot[0][2], BlueNoteSensorRotation.safe_sqrt(1-rot[0][2]*rot[0][2]))
            rot_z = math.atan2(-rot[0][1], rot[0][0])
        elif sequence == RotationSequence.XZY:
            rot_x = math.atan2(rot[2][1], rot[1][1])
            rot_z = math.atan2(-rot[0][1], BlueNoteSensorRotation.safe_sqrt(1-rot[0][1]*rot[0][1]))
            rot_y = math.atan2(rot[0][2], rot[0][0])
        elif sequence == RotationSequence.YXZ:
            rot_y = math.atan2(rot[0][2], rot[2][2])
            rot_x = math.atan2(-rot[1][2], BlueNoteSensorRotation.safe_sqrt(1-rot[1][2]*rot[1][2]))
            rot_z = math.atan2(rot[1][0], rot[1][1])
        elif sequence == RotationSequence.YZX:
            rot_y = math.atan2(-rot[2][0], rot[0][0])
            rot_z = math.atan2(rot[1][0], BlueNoteSensorRotation.safe_sqrt(1-rot[1][0]*rot[1][0]))
            rot_x = math.atan2(-rot[1][2], rot[1][1])
        elif sequence == RotationSequence.ZXY:
            rot_z = math.atan2(-rot[0][1], rot[1][1])
            rot_x = math.atan2(rot[2][1], BlueNoteSensorRotation.safe_sqrt(1-rot[2][1]*rot[2][1]))
            rot_y = math.atan2(-rot[2][0], rot[2][2])
        elif sequence == RotationSequence.ZYX:
            rot_z = math.atan2(rot[1][0], rot[0][0])
            rot_y = math.atan2(-rot[2][0], BlueNoteSensorRotation.safe_sqrt(1-rot[2][0]*rot[2][0]))
            rot_x = math.atan2(rot[2][1], rot[2][2])

        x_angle = rot_x * BlueNoteSensorRotation._rad_to_degree
        y_angle = rot_y * BlueNoteSensorRotation._rad_to_degree
        z_angle = rot_z * BlueNoteSensorRotation._rad_to_degree

        return x_angle, y_angle, z_angle

    @staticmethod
    def safe_sqrt(value):
        if value < 0 and -value < 1E-10:
            value = 0
        return math.sqrt(value)




