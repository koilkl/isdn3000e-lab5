#include <vector>
#include <string>
#include <cmath>

#include <Eigen/Geometry>
#include <igl/readOBJ.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/spatial/explog.hpp>
#include "imgui.h"




void task2() {

    polyscope::init();
    polyscope::view::setUpDir(polyscope::UpDir::ZUp);

    // Load URDF and compute initial forward kinematics
    std::string robot_dir = "robot/franka_description";
    std::string urdf_path = robot_dir + "/robot.urdf";
    pinocchio::Model model;
    pinocchio::GeometryModel geom_model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::urdf::buildGeom(model, urdf_path, pinocchio::VISUAL, geom_model, robot_dir);
    pinocchio::Data data(model);
    pinocchio::GeometryData geom_data(geom_model);
    Eigen::VectorXd q = pinocchio::neutral(model);
    Eigen::VectorXd q_init = q;
    std::vector<Eigen::MatrixXd> V_locals;
    std::vector<polyscope::SurfaceMesh*> meshes;
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::updateGeometryPlacements(model, data, geom_model, geom_data);
    // Gripper Control
    int finger1_q = model.joints[model.getJointId("fr3_finger_joint1")].idx_q();
    int finger2_q = model.joints[model.getJointId("fr3_finger_joint2")].idx_q();


    // Visualize the robot of the initial pose
    for (int i = 0; i < geom_model.geometryObjects.size(); ++i) {
        const auto& obj = geom_model.geometryObjects[i];
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readOBJ(obj.meshPath, V, F);
        pinocchio::SE3 M = geom_data.oMg[i];
        Eigen::Matrix3d R = M.rotation();
        Eigen::Vector3d t = M.translation();
        Eigen::MatrixXd Vw = (V * R.transpose()).rowwise() + t.transpose();
        auto* ms = polyscope::registerSurfaceMesh(obj.name + "_" + std::to_string(i), Vw, F);
        V_locals.push_back(V);
        meshes.push_back(ms);
    }

    // Visualize the target cube objects
    std::vector<std::array<double,3>> cubeV = {
        {-0.5,-0.5,-0.5},{ 0.5,-0.5,-0.5},{ 0.5, 0.5,-0.5},{-0.5, 0.5,-0.5},
        {-0.5,-0.5, 0.5},{ 0.5,-0.5, 0.5},{ 0.5, 0.5, 0.5},{-0.5, 0.5, 0.5}
    };
    std::vector<std::array<int,3>> cubeF = {
        {0,1,2},{0,2,3},
        {4,5,6},{4,6,7},
        {0,1,5},{0,5,4},
        {2,3,7},{2,7,6},
        {1,2,6},{1,6,5},
        {0,3,7},{0,7,4}
    };
    Eigen::Vector3d cube_t(0.55, 0.20, 0.60);
    double cube_size = 0.04;
    Eigen::Matrix3d cube_R =
        (Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(M_PI / 6.0, Eigen::Vector3d::UnitY())).toRotationMatrix();
    for (auto& v : cubeV) {
        Eigen::Vector3d p(v[0], v[1], v[2]);
        p = cube_size * (cube_R * p) + cube_t;
        v = {p.x(), p.y(), p.z()};
    }
    auto* cube = polyscope::registerSurfaceMesh("target_cube", cubeV, cubeF);


    // Define the start frame of robot end effector
    double axis_len = 0.20;
    std::vector<Eigen::Vector3d> ee_point(1), ee_x(1), ee_y(1), ee_z(1);
    pinocchio::FrameIndex ee_fid = model.getFrameId("fr3_hand_tcp");
    pinocchio::SE3 M_start = data.oMf[ee_fid];

    // Visualize the start frame of robot end effector
    ee_point[0] = M_start.translation();
    ee_x[0] = axis_len * M_start.rotation().col(0);
    ee_y[0] = axis_len * M_start.rotation().col(1);
    ee_z[0] = axis_len * M_start.rotation().col(2);
    auto* ee_cloud = polyscope::registerPointCloud("ee_frame", ee_point);
    ee_cloud->setPointRadius(0.02);
    auto* ee_x_q = ee_cloud->addVectorQuantity("x_axis", ee_x);
    ee_x_q->setVectorColor({1.0, 0.0, 0.0});
    ee_x_q->setVectorRadius(0.1);
    ee_x_q->setVectorLengthScale(0.1);
    ee_x_q->setEnabled(true);
    auto* ee_y_q = ee_cloud->addVectorQuantity("y_axis", ee_y);
    ee_y_q->setVectorColor({0.0, 1.0, 0.0});
    ee_y_q->setVectorRadius(0.1);
    ee_y_q->setVectorLengthScale(0.1);
    ee_y_q->setEnabled(true);
    auto* ee_z_q = ee_cloud->addVectorQuantity("z_axis", ee_z);
    ee_z_q->setVectorColor({0.0, 0.0, 1.0});
    ee_z_q->setVectorRadius(0.1);
    ee_z_q->setVectorLengthScale(0.1);
    ee_z_q->setEnabled(true);


    // Define the target frame
    std::vector<Eigen::Vector3d> target_point(1), target_x(1), target_y(1), target_z(1);
    Eigen::Matrix4d T_goal;
    // TODO 3: Revise the target frame to lead the robot to get the cube
    //  Hint: your can change the numbers of R and T_goal to obtain the correct frame
    Eigen::Matrix3d R =
        (Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(M_PI / 6.0, Eigen::Vector3d::UnitY())).toRotationMatrix();
    T_goal <<
        R(0,0), -R(0,1), -R(0,2), 0.55,
        R(1,0), -R(1,1), -R(1,2), 0.20,
        R(2,0), -R(2,1), -R(2,2), 0.60,
        0.0,    0.0,    0.0,    1.0;
    pinocchio::SE3 M_goal(
        T_goal.block<3,3>(0,0),
        T_goal.block<3,1>(0,3)
    );

    // Visualize the target frame
    target_point[0] = M_goal.translation();
    target_x[0] = axis_len * M_goal.rotation().col(0);
    target_y[0] = axis_len * M_goal.rotation().col(1);
    target_z[0] = axis_len * M_goal.rotation().col(2);
    auto* target_cloud = polyscope::registerPointCloud("target_frame", target_point);
    target_cloud->setPointRadius(0.025);
    auto* target_x_q = target_cloud->addVectorQuantity("x_axis", target_x);
    target_x_q->setVectorColor({1.0, 0.4, 0.4});
    target_x_q->setVectorRadius(0.1);
    target_x_q->setVectorLengthScale(0.1);
    target_x_q->setEnabled(true);
    auto* target_y_q = target_cloud->addVectorQuantity("y_axis", target_y);
    target_y_q->setVectorColor({0.4, 1.0, 0.4});
    target_y_q->setVectorRadius(0.1);
    target_y_q->setVectorLengthScale(0.1);
    target_y_q->setEnabled(true);
    auto* target_z_q = target_cloud->addVectorQuantity("z_axis", target_z);
    target_z_q->setVectorColor({0.4, 0.4, 1.0});
    target_z_q->setVectorRadius(0.1);
    target_z_q->setVectorLengthScale(0.1);
    target_z_q->setEnabled(true);



    Eigen::VectorXd q_start = q;
    Eigen::VectorXd q_goal = q_start;
    Eigen::VectorXd q_try = q_start;
    bool ik_success = false;
    const double alpha = 0.2;
    const double damping = 1e-3;
    const double eps = 1e-4;
    const int max_iters = 300;
    // TODO 1: Complete IK iteration
    //  We want the end-effector frame to match the target frame M_goal.
    //  In each iteration:
    //   1. Read the current end-effector pose from data.oMf[ee_fid]
    //   2. Compute the pose error from current EE frame to target frame
    //   3. Convert the SE3 error to a 6D twist vector using log6()
    //   4. Compute the frame Jacobian
    //   5. Solve a damped least-squares update
    //   6. Integrate the update into q_try
    for (int it = 0; it < max_iters; ++it) {
    pinocchio::framesForwardKinematics(model, data, q_try);
    pinocchio::SE3 M_ee = data.oMf[ee_fid];
    pinocchio::SE3 M_err = M_ee.inverse() * M_goal;
    Eigen::Matrix<double, 6, 1> err = pinocchio::log6(M_err).toVector();
    if (err.norm() < eps) {ik_success = true; q_goal = q_try; break;}
    Eigen::MatrixXd J(6, model.nv);
    pinocchio::computeFrameJacobian(model, data, q_try, ee_fid, pinocchio::LOCAL, J);
    Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::VectorXd dq = J.transpose() * (J * J.transpose() + damping * I).ldlt().solve(err);
    q_try = pinocchio::integrate(model, q_try, alpha * dq);
    if (finger1_q >= 0) q_try[finger1_q] = q_start[finger1_q];
    if (finger2_q >= 0) q_try[finger2_q] = q_start[finger2_q];
    }





    // UI callback function for IK, joint interpolation and gripper
    float s = 0.0f;
    float gripper_open = 0.04f;
    polyscope::state::userCallback = [&]() {
        ImGui::Text("Task5: IK + Joint Interpolation");
        ImGui::Separator();
        ImGui::SliderFloat("Motion", &s, 0.0f, 1.0f);
        if (ik_success) {
            // TODO 2: Joint interpolation
            //  The slider s is in [0, 1]:
            //    when s = 0, q should equal q_start
            //    when s = 1, q should equal q_goal
            //  So we linearly interpolate joints between q_start and q_goal via (1.0 - s) * q_start + s * q_goal
            q = (1.0 - s) * q_start + s * q_goal;

        } else {
            q = q_start;
            ImGui::TextColored(ImVec4(1,0,0,1), "IK failed: target not reachable.");
        }
        if (finger1_q >= 0) q[finger1_q] = gripper_open;
        if (finger2_q >= 0) q[finger2_q] = gripper_open;
        pinocchio::forwardKinematics(model, data, q);
        pinocchio::updateFramePlacements(model, data);
        pinocchio::updateGeometryPlacements(model, data, geom_model, geom_data);
        for (int i = 0; i < meshes.size(); ++i) {
            pinocchio::SE3 M = geom_data.oMg[i];
            Eigen::Matrix3d R = M.rotation();
            Eigen::Vector3d t = M.translation();
            Eigen::MatrixXd Vw = (V_locals[i] * R.transpose()).rowwise() + t.transpose();
            meshes[i]->updateVertexPositions(Vw);
        }
        pinocchio::SE3 M_ee = data.oMf[ee_fid];
        ee_point[0] = M_ee.translation();
        ee_x[0] = axis_len * M_ee.rotation().col(0);
        ee_y[0] = axis_len * M_ee.rotation().col(1);
        ee_z[0] = axis_len * M_ee.rotation().col(2);
        ee_cloud->updatePointPositions(ee_point);
        ee_x_q->updateData(ee_x);
        ee_y_q->updateData(ee_y);
        ee_z_q->updateData(ee_z);

        ImGui::SliderFloat("Gripper", &gripper_open, 0.0f, 0.04f);
        if (ImGui::Button("Reset")) {
            s = 0.0f;
            gripper_open = 0.04f;
        }
        ImGui::Text("IK success: %s", ik_success ? "true" : "false");
        ImGui::Text("s = %.3f", s);
    };

    polyscope::show();
}