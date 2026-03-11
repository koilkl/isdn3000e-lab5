#include <vector>
#include <string>
#include <array>
#include <iostream>

#include <Eigen/Dense>
#include <igl/readOBJ.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include <polyscope/point_cloud.h>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/geometry.hpp>

#include "imgui.h"




void task1() {

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
    float q_ui[7] = {0,0,0,0,0,0,0};
    const float q_min[7] = {-2.9f, -1.8f, -2.9f, -3.1f, -2.9f, -0.1f, -2.9f};
    const float q_max[7] = { 2.9f,  1.8f,  2.9f,  0.1f,  2.9f,  3.7f,  2.9f};
    std::vector<Eigen::MatrixXd> V_locals;
    std::vector<polyscope::SurfaceMesh*> meshes;
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::updateGeometryPlacements(model, data, geom_model, geom_data);

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

    // Define the current frame of the end effector
    double axis_len = 0.2;
    std::vector<Eigen::Vector3d> ee_point(1), ee_x(1), ee_y(1), ee_z(1);
    pinocchio::FrameIndex ee_fid = model.getFrameId("fr3_hand_tcp");
    const pinocchio::SE3& M = data.oMf[ee_fid];

    // Visualize the current frame of the end effector
    Eigen::Vector3d t = M.translation();
    Eigen::Matrix3d R = M.rotation();
    ee_point[0] = t;
    ee_x[0] = axis_len * R.col(0);
    ee_y[0] = axis_len * R.col(1);
    ee_z[0] = axis_len * R.col(2);

    // TODO 1: Draw the frame of the end effector, which consists of:
    //   - an origin point using ps_cloud = registerPointCloud()
    //   - three axis directions X, Y, Z using i.e. x_q = ps_cloud->addVectorQuantity("x_axis", ee_x)
    //      - set color to three axis directions: X = red, Y = green, Z = blue
    //      - set each direction vector radius to 0.1 and vector length scale to 0.1 to better visualize them
    //      - set three axis vectors enabled
    auto *ps_cloud = polyscope::registerPointCloud("ee_origin", ee_point);
    auto *x_q = ps_cloud->addVectorQuantity("x_axis", ee_x);
    auto *y_q = ps_cloud->addVectorQuantity("y_axis", ee_y);
    auto *z_q = ps_cloud->addVectorQuantity("z_axis", ee_z);
    x_q->setVectorColor({1.0, 0.0, 0.0});
    y_q->setVectorColor({0.0, 1.0, 0.0});
    z_q->setVectorColor({0.0, 0.0, 1.0});
    x_q->setVectorRadius(0.1);
    y_q->setVectorRadius(0.1);
    z_q->setVectorRadius(0.1);
    x_q->setVectorLengthScale(0.1);
    y_q->setVectorLengthScale(0.1);
    z_q->setVectorLengthScale(0.1);
    x_q->setEnabled(true);
    y_q->setEnabled(true);
    z_q->setEnabled(true);



    // UI callback of the meshes and the frames
    polyscope::state::userCallback = [&]() {
        ImGui::Text("Task3: Joint sliders");
        ImGui::Separator();
        for (int j = 0; j < 7; ++j) {
            std::string label = "q" + std::to_string(j);
            ImGui::SliderFloat(label.c_str(), &q_ui[j], q_min[j], q_max[j]);
        }
        for (int j = 0; j < 7; ++j)
            q[j] = q_ui[j];
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

        const pinocchio::SE3& M = data.oMf[ee_fid];
        Eigen::Vector3d t = M.translation();
        Eigen::Matrix3d R = M.rotation();
        ee_point[0] = t;
        ee_x[0] = axis_len * R.col(0);
        ee_y[0] = axis_len * R.col(1);
        ee_z[0] = axis_len * R.col(2);

        // TODO 2: Update the frame while callback, particularly, update the ps_cloud and three axis vector x_q, y_q and z_q
        ps_cloud->updatePointPositions(ee_point);
        x_q->updateData(ee_x);
        y_q->updateData(ee_y);
        z_q->updateData(ee_z);
        

    };

    polyscope::show();
}
