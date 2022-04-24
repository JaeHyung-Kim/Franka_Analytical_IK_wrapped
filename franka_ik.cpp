// Analytical Franka inverse kinematics using q7 as redundant parameter
// - Yanhao He, February 2020

#ifndef FRANKA_IK_HE_HPP
#define FRANKA_IK_HE_HPP

#include <array>
#include <cmath>
#include "eigen-3.4.0/Eigen/Dense"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


namespace py = pybind11;

// inverse kinematics w.r.t. End Effector Frame (using Franka Hand data)
std::array< std::array<double, 7>, 4 > franka_IK_EE ( const Eigen::Matrix3d &R_EE,
                                                      const Eigen::Vector3d &p_EE,
                                                      double q7,
                                                      const Eigen::Vector<double, 7> &q_actual )
{
    const std::array< std::array<double, 7>, 4 > q_all_NAN = {{ {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}},
                                                                {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}},
                                                                {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}},
                                                                {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}} }};
    const std::array<double, 7> q_NAN = {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}};
    std::array< std::array<double, 7>, 4 > q_all = q_all_NAN;
    
    // Eigen::Map< Eigen::Matrix<double, 4, 4> > O_T_EE(O_T_EE_array.data());
    
    const double d1 = 0.3330;
    const double d3 = 0.3160;
    const double d5 = 0.3840;
    const double d7e = 0.2104;
    const double a4 = 0.0825;
    const double a7 = 0.0880;
    
    const double LL24 = 0.10666225; // a4^2 + d3^2
    const double LL46 = 0.15426225; // a4^2 + d5^2
    const double L24 = 0.326591870689; // sqrt(LL24)
    const double L46 = 0.392762332715; // sqrt(LL46)
    
    const double thetaH46 = 1.35916951803; // atan(d5/a4);
    const double theta342 = 1.31542071191; // atan(d3/a4);
    const double theta46H = 0.211626808766; // acot(d5/a4);
    
    const std::array<double, 7> q_min = {{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
    const std::array<double, 7> q_max = {{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};
    
    if (q7 <= q_min[6] || q7 >= q_max[6])
        return q_all_NAN;
    else
        for (int i = 0; i < 4; i++)
            q_all[i][6] = q7;
    
    // compute p_6
    // Eigen::Matrix3d R_EE = O_T_EE.topLeftCorner<3, 3>();
    Eigen::Vector3d z_EE = R_EE.block<3, 1>(0, 2);
    // Eigen::Vector3d p_EE = O_T_EE.block<3, 1>(0, 3);
    Eigen::Vector3d p_7 = p_EE - d7e*z_EE;
    
    Eigen::Vector3d x_EE_6;
    x_EE_6 << std::cos(q7 - M_PI_4), -std::sin(q7 - M_PI_4), 0.0;
    Eigen::Vector3d x_6 = R_EE*x_EE_6;
    x_6 /= x_6.norm(); // visibly increases accuracy
    Eigen::Vector3d p_6 = p_7 - a7*x_6;
    
    // compute q4
    Eigen::Vector3d p_2;
    p_2 << 0.0, 0.0, d1;
    Eigen::Vector3d V26 = p_6 - p_2;
    
    double LL26 = V26[0]*V26[0] + V26[1]*V26[1] + V26[2]*V26[2];
    double L26 = std::sqrt(LL26);
    
    if (L24 + L46 < L26 || L24 + L26 < L46 || L26 + L46 < L24)
        return q_all_NAN;
    
    double theta246 = std::acos((LL24 + LL46 - LL26)/2.0/L24/L46);
    double q4 = theta246 + thetaH46 + theta342 - 2.0*M_PI;
    if (q4 <= q_min[3] || q4 >= q_max[3])
        return q_all_NAN;
    else
        for (int i = 0; i < 4; i++)
            q_all[i][3] = q4;
    
    // compute q6
    double theta462 = std::acos((LL26 + LL46 - LL24)/2.0/L26/L46);
    double theta26H = theta46H + theta462;
    double D26 = -L26*std::cos(theta26H);
    
    Eigen::Vector3d Z_6 = z_EE.cross(x_6);
    Eigen::Vector3d Y_6 = Z_6.cross(x_6);
    Eigen::Matrix3d R_6;
    R_6.col(0) = x_6;
    R_6.col(1) = Y_6/Y_6.norm();
    R_6.col(2) = Z_6/Z_6.norm();
    Eigen::Vector3d V_6_62 = R_6.transpose()*(-V26);

    double Phi6 = std::atan2(V_6_62[1], V_6_62[0]);
    double Theta6 = std::asin(D26/std::sqrt(V_6_62[0]*V_6_62[0] + V_6_62[1]*V_6_62[1]));
    
    std::array<double, 2> q6;
    q6[0] = M_PI - Theta6 - Phi6;
    q6[1] = Theta6 - Phi6;
    
    for (int i = 0; i < 2; i++)
    {
        if (q6[i] <= q_min[5])
            q6[i] += 2.0*M_PI;
        else if (q6[i] >= q_max[5])
            q6[i] -= 2.0*M_PI;
        
        if (q6[i] <= q_min[5] || q6[i] >= q_max[5])
        {
            q_all[2*i] = q_NAN;
            q_all[2*i + 1] = q_NAN;
        }
        else
        {
            q_all[2*i][5] = q6[i];
            q_all[2*i + 1][5] = q6[i];
        }
    }
    if (std::isnan(q_all[0][5]) && std::isnan(q_all[2][5]))
        return q_all_NAN;

    // compute q1 & q2
    double thetaP26 = 3.0*M_PI_2 - theta462 - theta246 - theta342;
    double thetaP = M_PI - thetaP26 - theta26H;
    double LP6 = L26*sin(thetaP26)/std::sin(thetaP);
    
    std::array< Eigen::Vector3d, 4 > z_5_all;
    std::array< Eigen::Vector3d, 4 > V2P_all;
    
    for (int i = 0; i < 2; i++)
    {
        Eigen::Vector3d z_6_5;
        z_6_5 << std::sin(q6[i]), std::cos(q6[i]), 0.0;
        Eigen::Vector3d z_5 = R_6*z_6_5;
        Eigen::Vector3d V2P = p_6 - LP6*z_5 - p_2;
        
        z_5_all[2*i] = z_5;
        z_5_all[2*i + 1] = z_5;
        V2P_all[2*i] = V2P;
        V2P_all[2*i + 1] = V2P;
        
        double L2P = V2P.norm();
        
        if (std::fabs(V2P[2]/L2P) > 0.999)
        {
            q_all[2*i][0] = q_actual(0);
            q_all[2*i][1] = 0.0;
            q_all[2*i + 1][0] = q_actual(0);
            q_all[2*i + 1][1] = 0.0;
        }
        else
        {
            q_all[2*i][0] = std::atan2(V2P[1], V2P[0]);
            q_all[2*i][1] = std::acos(V2P[2]/L2P);
            if (q_all[2*i][0] < 0)
                q_all[2*i + 1][0] = q_all[2*i][0] + M_PI;
            else
                q_all[2*i + 1][0] = q_all[2*i][0] - M_PI;
            q_all[2*i + 1][1] = -q_all[2*i][1];
        }
    }
    
    for (int i = 0; i < 4; i++)
    {
        if ( q_all[i][0] <= q_min[0] || q_all[i][0] >= q_max[0]
             || q_all[i][1] <= q_min[1] || q_all[i][1] >= q_max[1] )
        {
            q_all[i] = q_NAN;
            continue;
        }

        // compute q3
        Eigen::Vector3d z_3 = V2P_all[i]/V2P_all[i].norm();
        Eigen::Vector3d Y_3 = -V26.cross(V2P_all[i]);
        Eigen::Vector3d y_3 = Y_3/Y_3.norm();
        Eigen::Vector3d x_3 = y_3.cross(z_3);
        Eigen::Matrix3d R_1;
        double c1 = std::cos(q_all[i][0]);
        double s1 = std::sin(q_all[i][0]);
        R_1 <<   c1,  -s1,  0.0,
                 s1,   c1,  0.0,
                0.0,  0.0,  1.0;
        Eigen::Matrix3d R_1_2;
        double c2 = std::cos(q_all[i][1]);
        double s2 = std::sin(q_all[i][1]);
        R_1_2 <<   c2,  -s2, 0.0,
                  0.0,  0.0, 1.0,
                  -s2,  -c2, 0.0;
        Eigen::Matrix3d R_2 = R_1*R_1_2;
        Eigen::Vector3d x_2_3 = R_2.transpose()*x_3;
        q_all[i][2] = std::atan2(x_2_3[2], x_2_3[0]);
        
        if (q_all[i][2] <= q_min[2] || q_all[i][2] >= q_max[2])
        {
            q_all[i] = q_NAN;
            continue;
        }
        
        // compute q5
        Eigen::Vector3d VH4 = p_2 + d3*z_3 + a4*x_3 - p_6 + d5*z_5_all[i];
        Eigen::Matrix3d R_5_6;
        double c6 = std::cos(q_all[i][5]);
        double s6 = std::sin(q_all[i][5]);
        R_5_6 <<   c6,  -s6,  0.0,
                  0.0,  0.0, -1.0,
                   s6,   c6,  0.0;
        Eigen::Matrix3d R_5 = R_6*R_5_6.transpose();
        Eigen::Vector3d V_5_H4 = R_5.transpose()*VH4;
        
        q_all[i][4] = -std::atan2(V_5_H4[1], V_5_H4[0]);
        if (q_all[i][4] <= q_min[4] || q_all[i][4] >= q_max[4])
        {
            q_all[i] = q_NAN;
            continue;
        }
    }
    
    return q_all;
}

// "Case-Consistent" inverse kinematics w.r.t. End Effector Frame (using Franka Hand data)
std::array<double, 7> franka_IK_EE_CC ( std::array<double, 16> O_T_EE_array,
                                        double q7,
                                        std::array<double, 7> q_actual_array )
{
    const std::array<double, 7> q_NAN = {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}};
    
    std::array<double, 7> q;
    
    Eigen::Map< Eigen::Matrix<double, 4, 4> > O_T_EE(O_T_EE_array.data());
    
    // constants
    const double d1 = 0.3330;
    const double d3 = 0.3160;
    const double d5 = 0.3840;
    const double d7e = 0.2104;
    const double a4 = 0.0825;
    const double a7 = 0.0880;

    const double LL24 = 0.10666225; // a4^2 + d3^2
    const double LL46 = 0.15426225; // a4^2 + d5^2
    const double L24 = 0.326591870689; // sqrt(LL24)
    const double L46 = 0.392762332715; // sqrt(LL46)
    
    const double thetaH46 = 1.35916951803; // atan(d5/a4);
    const double theta342 = 1.31542071191; // atan(d3/a4);
    const double theta46H = 0.211626808766; // acot(d5/a4);
    
    const std::array<double, 7> q_min = {{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
    const std::array<double, 7> q_max = {{ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973}};
    
    // return NAN if input q7 is out of range
    if (q7 <= q_min[6] || q7 >= q_max[6])
        return q_NAN;
    else
        q[6] = q7;

    // FK for getting current case id
    double c1_a = std::cos(q_actual_array[0]); double s1_a = std::sin(q_actual_array[0]);
    double c2_a = std::cos(q_actual_array[1]); double s2_a = std::sin(q_actual_array[1]);
    double c3_a = std::cos(q_actual_array[2]); double s3_a = std::sin(q_actual_array[2]);
    double c4_a = std::cos(q_actual_array[3]); double s4_a = std::sin(q_actual_array[3]);
    double c5_a = std::cos(q_actual_array[4]); double s5_a = std::sin(q_actual_array[4]);
    double c6_a = std::cos(q_actual_array[5]); double s6_a = std::sin(q_actual_array[5]);

    std::array< Eigen::Matrix<double, 4, 4>, 7> As_a;
    As_a[0] <<   c1_a, -s1_a,  0.0,  0.0,    // O1
                 s1_a,  c1_a,  0.0,  0.0,
                  0.0,   0.0,  1.0,   d1,
                  0.0,   0.0,  0.0,  1.0;
    As_a[1] <<   c2_a, -s2_a,  0.0,  0.0,    // O2
                  0.0,   0.0,  1.0,  0.0,
                -s2_a, -c2_a,  0.0,  0.0,
                  0.0,   0.0,  0.0,  1.0;
    As_a[2] <<   c3_a, -s3_a,  0.0,  0.0,    // O3
                  0.0,   0.0, -1.0,  -d3,
                 s3_a,  c3_a,  0.0,  0.0,
                  0.0,   0.0,  0.0,  1.0;
    As_a[3] <<   c4_a, -s4_a,  0.0,   a4,    // O4
                  0.0,   0.0, -1.0,  0.0,
                 s4_a,  c4_a,  0.0,  0.0,
                  0.0,   0.0,  0.0,  1.0;
    As_a[4] <<    1.0,   0.0,  0.0,  -a4,    // H
                  0.0,   1.0,  0.0,  0.0,
                  0.0,   0.0,  1.0,  0.0,
                  0.0,   0.0,  0.0,  1.0;
    As_a[5] <<   c5_a, -s5_a,  0.0,  0.0,    // O5
                  0.0,   0.0,  1.0,   d5,
                -s5_a, -c5_a,  0.0,  0.0,
                  0.0,   0.0,  0.0,  1.0;
    As_a[6] <<   c6_a, -s6_a,  0.0,  0.0,    // O6
                  0.0,   0.0, -1.0,  0.0,
                 s6_a,  c6_a,  0.0,  0.0,
                  0.0,   0.0,  0.0,  1.0;
    std::array< Eigen::Matrix<double, 4, 4>, 7> Ts_a;
    Ts_a[0] = As_a[0];
    for (unsigned int j = 1; j < 7; j++)
        Ts_a[j] = Ts_a[j - 1]*As_a[j];

    // identify q6 case
    Eigen::Vector3d V62_a = Ts_a[1].block<3, 1>(0, 3) - Ts_a[6].block<3, 1>(0, 3);
    Eigen::Vector3d V6H_a = Ts_a[4].block<3, 1>(0, 3) - Ts_a[6].block<3, 1>(0, 3);
    Eigen::Vector3d Z6_a = Ts_a[6].block<3, 1>(0, 2);
    bool is_case6_0 = ((V6H_a.cross(V62_a)).transpose()*Z6_a <= 0);

    // identify q1 case
    bool is_case1_1 = (q_actual_array[1] < 0);
    
    // IK: compute p_6
    Eigen::Matrix3d R_EE = O_T_EE.topLeftCorner<3, 3>();
    Eigen::Vector3d z_EE = O_T_EE.block<3, 1>(0, 2);
    Eigen::Vector3d p_EE = O_T_EE.block<3, 1>(0, 3);
    Eigen::Vector3d p_7 = p_EE - d7e*z_EE;
    
    Eigen::Vector3d x_EE_6;
    x_EE_6 << std::cos(q7 - M_PI_4), -std::sin(q7 - M_PI_4), 0.0;
    Eigen::Vector3d x_6 = R_EE*x_EE_6;
    x_6 /= x_6.norm(); // visibly increases accuracy
    Eigen::Vector3d p_6 = p_7 - a7*x_6;
    
    // IK: compute q4
    Eigen::Vector3d p_2;
    p_2 << 0.0, 0.0, d1;
    Eigen::Vector3d V26 = p_6 - p_2;
    
    double LL26 = V26[0]*V26[0] + V26[1]*V26[1] + V26[2]*V26[2];
    double L26 = std::sqrt(LL26);
    
    if (L24 + L46 < L26 || L24 + L26 < L46 || L26 + L46 < L24)
        return q_NAN;
    
    double theta246 = std::acos((LL24 + LL46 - LL26)/2.0/L24/L46);
    q[3] = theta246 + thetaH46 + theta342 - 2.0*M_PI;
    if (q[3] <= q_min[3] || q[3] >= q_max[3])
        return q_NAN;
    
    // IK: compute q6
    double theta462 = std::acos((LL26 + LL46 - LL24)/2.0/L26/L46);
    double theta26H = theta46H + theta462;
    double D26 = -L26*std::cos(theta26H);
    
    Eigen::Vector3d Z_6 = z_EE.cross(x_6);
    Eigen::Vector3d Y_6 = Z_6.cross(x_6);
    Eigen::Matrix3d R_6;
    R_6.col(0) = x_6;
    R_6.col(1) = Y_6/Y_6.norm();
    R_6.col(2) = Z_6/Z_6.norm();
    Eigen::Vector3d V_6_62 = R_6.transpose()*(-V26);

    double Phi6 = std::atan2(V_6_62[1], V_6_62[0]);
    double Theta6 = std::asin(D26/std::sqrt(V_6_62[0]*V_6_62[0] + V_6_62[1]*V_6_62[1]));
    
    if (is_case6_0)
        q[5] = M_PI - Theta6 - Phi6;
    else
        q[5] = Theta6 - Phi6;
    
    if (q[5] <= q_min[5])
        q[5] += 2.0*M_PI;
    else if (q[5] >= q_max[5])
        q[5] -= 2.0*M_PI;
    
    if (q[5] <= q_min[5] || q[5] >= q_max[5])
        return q_NAN;

    // IK: compute q1 & q2
    double thetaP26 = 3.0*M_PI_2 - theta462 - theta246 - theta342;
    double thetaP = M_PI - thetaP26 - theta26H;
    double LP6 = L26*sin(thetaP26)/std::sin(thetaP);
    
    Eigen::Vector3d z_6_5;
    z_6_5 << std::sin(q[5]), std::cos(q[5]), 0.0;
    Eigen::Vector3d z_5 = R_6*z_6_5;
    Eigen::Vector3d V2P = p_6 - LP6*z_5 - p_2;
    
    double L2P = V2P.norm();
    
    if (std::fabs(V2P[2]/L2P) > 0.999)
    {
        q[0] = q_actual_array[0];
        q[1] = 0.0;
    }
    else
    {
        q[0] = std::atan2(V2P[1], V2P[0]);
        q[1] = std::acos(V2P[2]/L2P);
        if (is_case1_1)
        {
            if (q[0] < 0.0)
                q[0] += M_PI;
            else
                q[0] -= M_PI;
            q[1] = -q[1];
        }
    }
    
    if ( q[0] <= q_min[0] || q[0] >= q_max[0]
      || q[1] <= q_min[1] || q[1] >= q_max[1] )
        return q_NAN;
    
    // IK: compute q3
    Eigen::Vector3d z_3 = V2P/V2P.norm();
    Eigen::Vector3d Y_3 = -V26.cross(V2P);
    Eigen::Vector3d y_3 = Y_3/Y_3.norm();
    Eigen::Vector3d x_3 = y_3.cross(z_3);
    Eigen::Matrix3d R_1;
    double c1 = std::cos(q[0]);
    double s1 = std::sin(q[0]);
    R_1 <<   c1,  -s1,  0.0,
             s1,   c1,  0.0,
            0.0,  0.0,  1.0;
    Eigen::Matrix3d R_1_2;
    double c2 = std::cos(q[1]);
    double s2 = std::sin(q[1]);
    R_1_2 <<   c2,  -s2, 0.0,
              0.0,  0.0, 1.0,
              -s2,  -c2, 0.0;
    Eigen::Matrix3d R_2 = R_1*R_1_2;
    Eigen::Vector3d x_2_3 = R_2.transpose()*x_3;
    q[2] = std::atan2(x_2_3[2], x_2_3[0]);
    
    if (q[2] <= q_min[2] || q[2] >= q_max[2])
        return q_NAN;
    
    // IK: compute q5
    Eigen::Vector3d VH4 = p_2 + d3*z_3 + a4*x_3 - p_6 + d5*z_5;
    Eigen::Matrix3d R_5_6;
    double c6 = std::cos(q[5]);
    double s6 = std::sin(q[5]);
    R_5_6 <<   c6,  -s6,  0.0,
              0.0,  0.0, -1.0,
               s6,   c6,  0.0;
    Eigen::Matrix3d R_5 = R_6*R_5_6.transpose();
    Eigen::Vector3d V_5_H4 = R_5.transpose()*VH4;
    
    q[4] = -std::atan2(V_5_H4[1], V_5_H4[0]);
    if (q[4] <= q_min[4] || q[4] >= q_max[4])
        return q_NAN;
    
    return q;
}

void franka_IK_Nearest(
    const Eigen::Matrix3d &targetRotation,
    const Eigen::Vector3d &targetPosition,
    double q7,
    const Eigen::Vector<double, 7> &currentPosition,
    Eigen::Ref<Eigen::Vector<double, 10>> jointPosition
)
{
    std::array<std::array<double, 7>, 4> resultArray = franka_IK_EE(targetRotation, targetPosition, q7, currentPosition);
    bool feasible;
    double minimumDistance = -1.0;
    int index = -1;
    for (int i = 0; i < 4; i++)
    {
        feasible = true;
        for (int j = 0; j < 7; j++)
        {
            if (std::isnan(resultArray[i][j]))
            {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;
        Eigen::Map<Eigen::Vector<double, 7>> result(resultArray[i].data());
        double distance = (currentPosition - result).norm();
        if ((minimumDistance < 0) || (distance < minimumDistance))
        {
            minimumDistance = distance;
            index = i;
        }
    }
    if (index >= 0)
    {
        for (int i = 0; i < 7; i++) jointPosition(i) = resultArray[index][i];
        jointPosition(9) = 1;
    }
}

Eigen::Vector<double, 10> franka_IK_q7(
    Eigen::Ref<const Eigen::Vector3d> targetHandPosition, 
    Eigen::Ref<const Eigen::Vector4d> targetHandOrientation, 
    double width, 
    double q7
)
{
    Eigen::Quaterniond q(targetHandOrientation);
    Eigen::Matrix3d targetRotation = q.normalized().toRotationMatrix();
    Eigen::Vector3d offset(0.0, 0.0, 0.1034);
    Eigen::Vector3d targetPosition = targetRotation * offset + targetHandPosition;
    Eigen::Vector<double, 7> currentPosition = {0.0000, 0.0000, 0.0000, -0.9425, 0.0000, 1.1205, 0.0000};

    std::array<std::array<double, 7>, 4> resultArray = franka_IK_EE(targetRotation, targetPosition, q7, currentPosition);

    Eigen::Vector<double, 10> jointPosition;
    jointPosition.setZero();

    bool feasible;
    double minimumDistance = -1.0;
    int index = -1;
    for (int i = 0; i < 4; i++)
    {
        feasible = true;
        for (int j = 0; j < 7; j++)
        {
            if (std::isnan(resultArray[i][j]))
            {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;
        Eigen::Map<Eigen::Vector<double, 7>> result(resultArray[i].data());
        double distance = (currentPosition - result).norm();
        if ((minimumDistance < 0) || (distance < minimumDistance))
        {
            minimumDistance = distance;
            index = i;
        }
    }
    if (index >= 0)
    {
        for (int i = 0; i < 7; i++) jointPosition(i) = resultArray[index][i];
        jointPosition(7) = jointPosition(8) = width;
        jointPosition(9) = 1;
    }

    return jointPosition;
}

Eigen::Vector<double, 10> franka_IK(
    Eigen::Ref<const Eigen::Vector3d> targetHandPosition, 
    Eigen::Ref<const Eigen::Vector4d> targetHandOrientation, 
    double width
)
{
    Eigen::Quaterniond q(targetHandOrientation);
    Eigen::Matrix3d targetRotation = q.normalized().toRotationMatrix();
    Eigen::Vector3d offset(0.0, 0.0, 0.1034);
    Eigen::Vector3d targetPosition = targetRotation * offset + targetHandPosition;
    Eigen::Vector<double, 7> currentPosition = {0.0000, 0.0000, 0.0000, -0.9425, 0.0000, 1.1205, 0.0000};
    Eigen::Vector<double, 41> q7Candidates = Eigen::VectorXd::LinSpaced(43, -2.8973, 2.8973)(Eigen::seq(1, 41));
    Eigen::Matrix<double, 10, 41> jointPositions;
    int i;
    int index = -1;
    double difference;
    double minimumDifference = -1.0;

    jointPositions.setZero();
    for (i = 0; i < 41; i++)
    {
        franka_IK_Nearest(targetRotation, targetPosition, q7Candidates(i), currentPosition, jointPositions.col(i));
    }

    for (i = 0; i < 41; i++)
    {
        if (!(jointPositions(9, i) > 0)) continue;
        difference = (currentPosition - jointPositions(Eigen::seq(0, 6), i)).norm();
        if ((minimumDifference < 0) || (difference < minimumDifference))
        {
            minimumDifference = difference;
            index = i;
        }
    }
    if (index >= 0) jointPositions(7, index) = jointPositions(8, index) = width;
    else index = 0;
    return jointPositions.col(index);
}

Eigen::Vector<double, 10> franka_IK_2(
    Eigen::Ref<const Eigen::Vector3d> targetHandPosition, 
    Eigen::Ref<const Eigen::Vector4d> targetHandOrientation, 
    double width
)
{
    Eigen::Quaterniond q(targetHandOrientation);
    Eigen::Matrix3d targetRotation = q.normalized().toRotationMatrix();
    Eigen::Vector3d offset(0.0, 0.0, 0.1034);
    Eigen::Vector3d targetPosition = targetRotation * offset + targetHandPosition;
    Eigen::Vector<double, 7> currentPosition = {0.0000, 0.0000, 0.0000, -0.9425, 0.0000, 1.1205, 0.0000};
    Eigen::Vector<double, 41> q7Candidates = Eigen::VectorXd::LinSpaced(43, -2.8973, 2.8973)(Eigen::seq(1, 41));
    Eigen::Matrix<double, 10, 41> jointPositions;
    int i;
    int index = -1;
    double difference;
    double minimumDifference = -1.0;

    jointPositions.setZero();
    for (i = 0; i < 41; i++)
    {
        franka_IK_Nearest(targetRotation, targetPosition, q7Candidates(i), currentPosition, jointPositions.col(i));
    }

    for (i = 0; i < 41; i++)
    {
        if (!(jointPositions(9, i) > 0)) continue;
        difference = (currentPosition - jointPositions(Eigen::seq(0, 6), i)).norm();
        if ((minimumDifference < 0) || (difference < minimumDifference))
        {
            minimumDifference = difference;
            index = i;
        }
    }
    if (index >= 0) jointPositions(7, index) = jointPositions(8, index) = width;
    else index = 0;
    return jointPositions.col(index);
}

PYBIND11_MODULE(franka_ik, m)
{
    m.doc() = "Analytical IK";
    m.def("franka_IK_q7", &franka_IK_q7);
    m.def("franka_IK", &franka_IK);
    m.def("franka_IK_2", &franka_IK_2);
}

#endif // FRANKA_IK_HE_HPP

// c++ -O3 -Wall -shared -std=c++11 -fPIC -I/home/user/Workspace/Franka_Analytical_IK_wrapped/eigen-3.4.0 $(python3 -m pybind11 --includes) franka_ik.cpp -o franka_ik$(python3-config --extension-suffix)
