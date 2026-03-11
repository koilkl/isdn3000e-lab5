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

void task3() {

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

    // Visualize an obstacle wall
    std::vector<std::array<double,3>> wallV = {
        {-0.5,-0.5,-0.5},{ 0.5,-0.5,-0.5},{ 0.5, 0.5,-0.5},{-0.5, 0.5,-0.5},
        {-0.5,-0.5, 0.5},{ 0.5,-0.5, 0.5},{ 0.5, 0.5, 0.5},{-0.5, 0.5, 0.5}
    };
    std::vector<std::array<int,3>> wallF = {
        {0,1,2},{0,2,3},
        {4,5,6},{4,6,7},
        {0,1,5},{0,5,4},
        {2,3,7},{2,7,6},
        {1,2,6},{1,6,5},
        {0,3,7},{0,7,4}
    };
    Eigen::Vector3d wall_t(0.42, 0.25, 0.65);
    Eigen::Vector3d wall_scale(0.02, 0.35, 0.22);
    Eigen::Matrix3d wall_R = Eigen::Matrix3d::Identity();
    for (auto& v : wallV) {
        Eigen::Vector3d p(v[0], v[1], v[2]);
        p = wall_R * wall_scale.asDiagonal() * p + wall_t;
        v = {p.x(), p.y(), p.z()};
    }
    auto* wall = polyscope::registerSurfaceMesh("obstacle_wall", wallV, wallF);

    // Define the start frame of robot end effector
    double axis_len = 0.20;
    std::vector<Eigen::Vector3d> ee_point(1), ee_x(1), ee_y(1), ee_z(1);
    pinocchio::FrameIndex ee_fid = model.getFrameId("fr3_hand_tcp");
    pinocchio::SE3 M_start = data.oMf[ee_fid];

    // Define the start frame of robot end effector
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

    // Helper for frame visualization
    auto register_frame = [&](const std::string& name, const pinocchio::SE3& M,
                              double point_radius,
                              std::array<double,3> x_color,
                              std::array<double,3> y_color,
                              std::array<double,3> z_color) {
        std::vector<Eigen::Vector3d> p(1), x(1), y(1), z(1);
        p[0] = M.translation();
        x[0] = axis_len * M.rotation().col(0);
        y[0] = axis_len * M.rotation().col(1);
        z[0] = axis_len * M.rotation().col(2);
        auto* cloud = polyscope::registerPointCloud(name, p);
        cloud->setPointRadius(point_radius);
        auto* x_q = cloud->addVectorQuantity(name + "_x_axis", x);
        x_q->setVectorColor({x_color[0], x_color[1], x_color[2]});
        x_q->setVectorRadius(0.1);
        x_q->setVectorLengthScale(0.1);
        x_q->setEnabled(true);
        auto* y_q = cloud->addVectorQuantity(name + "_y_axis", y);
        y_q->setVectorColor({y_color[0], y_color[1], y_color[2]});
        y_q->setVectorRadius(0.1);
        y_q->setVectorLengthScale(0.1);
        y_q->setEnabled(true);
        auto* z_q = cloud->addVectorQuantity(name + "_z_axis", z);
        z_q->setVectorColor({z_color[0], z_color[1], z_color[2]});
        z_q->setVectorRadius(0.1);
        z_q->setVectorLengthScale(0.1);
        z_q->setEnabled(true);
    };

    // Helper IK solver for one target frame
    auto solve_ik = [&](const Eigen::VectorXd& q_seed,
                        const pinocchio::SE3& M_goal,
                        Eigen::VectorXd& q_sol) {
        Eigen::VectorXd q_try = q_seed;
        bool ik_success = false;
        const double alpha = 0.2;
        const double damping = 1e-3;
        const double eps = 1e-4;
        const int max_iters = 300;
        for (int it = 0; it < max_iters; ++it) {
            pinocchio::framesForwardKinematics(model, data, q_try);
            pinocchio::SE3 M_ee = data.oMf[ee_fid];
            pinocchio::SE3 M_err = M_ee.inverse() * M_goal;
            Eigen::Matrix<double, 6, 1> err = pinocchio::log6(M_err).toVector();
            if (err.norm() < eps) {
                ik_success = true;
                q_sol = q_try;
                break;
            }
            Eigen::MatrixXd J(6, model.nv);
            pinocchio::computeFrameJacobian(model, data, q_try, ee_fid, pinocchio::LOCAL, J);
            Eigen::Matrix<double, 6, 6> A =
                J * J.transpose() + damping * Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::VectorXd dq = J.transpose() * A.ldlt().solve(err);
            q_try = pinocchio::integrate(model, q_try, alpha * dq);
            if (finger1_q >= 0) q_try[finger1_q] = q_seed[finger1_q];
            if (finger2_q >= 0) q_try[finger2_q] = q_seed[finger2_q];
        }
        return ik_success;
    };

    std::vector<Eigen::Matrix4d> T_waypoints;
    // -------------------------- Modified Waypoints --------------------------
    // Waypoint 1: Initial safe position (left of obstacle, avoid x=0.40~0.44)
    // Position: (0.30, 0.0, 0.70) | Rotation: Match cube's orientation
    Eigen::Matrix4d T_waypoint1;
    T_waypoint1 <<
        -cube_R(0,0), -cube_R(0,1), cube_R(0,2), 0.30,
        -cube_R(1,0), -cube_R(1,1), cube_R(1,2), 0.00,
        -cube_R(2,0), -cube_R(2,1), cube_R(2,2), 0.70,
        0.0,         0.0,         0.0,         1.0;

    // Waypoint 2: Detour around obstacle (right side, x=0.45 > 0.44)
    // Position: (0.45, 0.10, 0.70) | Rotation: Keep cube's orientation
    Eigen::Matrix4d T_waypoint2;
    T_waypoint2 <<
        cube_R(0,0), cube_R(0,1), cube_R(0,2), 0.45,
        cube_R(1,0), cube_R(1,1), cube_R(1,2), 0.00,
        cube_R(2,0), cube_R(2,1), cube_R(2,2), 0.70,
        0.0,         0.0,         0.0,         1.0;

    // Waypoint 3: Above the target cube (z=0.70 > cube's z=0.60)
    // Position: (0.55, 0.20, 0.70) | Rotation: Exact cube orientation
    Eigen::Matrix4d T_waypoint3;
    T_waypoint3 <<
        cube_R(0,0), cube_R(0,1), cube_R(0,2), 0.55,
        cube_R(1,0), cube_R(1,1), cube_R(1,2), 0.20,
        cube_R(2,0), cube_R(2,1), cube_R(2,2), 0.50,
        0.0,         0.0,         0.0,         1.0;

    // Waypoint 4 (Goal): Exact cube position (gripping pose)
    // Position: (0.55, 0.20, 0.60) | Rotation: Exact cube orientation
    Eigen::Matrix4d T_goal;
    T_goal <<
        cube_R(0,0), cube_R(0,1), cube_R(0,2), 0.55,
        cube_R(1,0), cube_R(1,1), cube_R(1,2), 0.20,
        cube_R(2,0), cube_R(2,1), cube_R(2,2), 0.60,
        0.0,         0.0,         0.0,         1.0;
    // ------------------------------------------------------------------------

    // Add all waypoints to the path
    T_waypoints.push_back(T_waypoint1);
    T_waypoints.push_back(T_waypoint2);
    T_waypoints.push_back(T_waypoint3);
    T_waypoints.push_back(T_goal);

    // Visualize waypoint frames
    for (int i = 0; i < T_waypoints.size(); ++i) {
        pinocchio::SE3 M_wp(
            T_waypoints[i].block<3,3>(0,0),
            T_waypoints[i].block<3,1>(0,3)
        );
        register_frame(
            "waypoint_frame_" + std::to_string(i),
            M_wp,
            0.02,
            {1.0, 0.6, 0.6},
            {0.6, 1.0, 0.6},
            {0.6, 0.6, 1.0}
        );
    }

    // Solve IK waypoint by waypoint
    Eigen::VectorXd q_start = q;
    std::vector<Eigen::VectorXd> q_path;
    std::vector<bool> waypoint_success;
    q_path.push_back(q_start);
    Eigen::VectorXd q_curr = q_start;
    bool all_ik_success = true;
    for (int i = 0; i < T_waypoints.size(); ++i) {
        pinocchio::SE3 M_goal_i(
            T_waypoints[i].block<3,3>(0,0),
            T_waypoints[i].block<3,1>(0,3)
        );
        Eigen::VectorXd q_next = q_curr;
        bool ok = solve_ik(q_curr, M_goal_i, q_next);
        waypoint_success.push_back(ok);
        if (!ok) {
            all_ik_success = false;
            break;
        }
        q_path.push_back(q_next);
        q_curr = q_next;
    }

    float s = 0.0f;
    float gripper_open = 0.04f;
    polyscope::state::userCallback = [&]() {
        ImGui::Text("Task3: Multi-Frame IK Path");
        ImGui::Separator();
        ImGui::SliderFloat("Motion", &s, 0.0f, 1.0f);
        if (all_ik_success && q_path.size() >= 2) {
            int num_segments = q_path.size() - 1;
            float u = s * num_segments;
            int seg = std::min((int)std::floor(u), num_segments - 1);
            float local_s = u - seg;
            q = (1.0 - local_s) * q_path[seg] + local_s * q_path[seg + 1];
            ImGui::Text("Current segment: %d / %d", seg + 1, num_segments);
            ImGui::Text("Local segment s: %.3f", local_s);
        } else {
            q = q_start;
            ImGui::TextColored(ImVec4(1,0,0,1), "Path IK failed: at least one waypoint is not reachable.");
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
        ImGui::Text("All path IK success: %s", all_ik_success ? "true" : "false");
        for (int i = 0; i < waypoint_success.size(); ++i) {
            ImGui::Text("Waypoint %d IK: %s", i + 1, waypoint_success[i] ? "true" : "false");
        }
        ImGui::Text("s = %.3f", s);
    };

    polyscope::show();
}