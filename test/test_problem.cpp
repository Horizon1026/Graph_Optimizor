#include <iostream>
#include <random>

#include <include/ba_solver/graph_optimizor/problem.hpp>
#include <vertex_pose.hpp>
#include <vertex_landmark_pos.hpp>
#include <edge_reprojection_pos.hpp>

using Scalar = float;      // 方便切换浮点数精度

/*
 * Frame : 保存每帧的姿态和观测
 */
struct Frame {
    Frame(GraphOptimizor::Matrix3<Scalar> R, GraphOptimizor::Vector3<Scalar> t) : Rwc(R), qwc(R), twc(t) {};
    GraphOptimizor::Matrix3<Scalar> Rwc;
    GraphOptimizor::Quaternion<Scalar> qwc;
    GraphOptimizor::Vector3<Scalar> twc;

    std::unordered_map<int, GraphOptimizor::Vector3<Scalar>> featurePerId; // 该帧观测到的特征以及特征id
};


/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */
void GetSimDataInWordFrame(std::vector<Frame> &cameraPoses, std::vector<GraphOptimizor::Vector3<Scalar>> &points) {
    int featureNums = 5;  // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 3;      // 相机数目

    double radius = 8;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 16); // 1/16 圆弧
        // 绕 z 轴 旋转
        GraphOptimizor::Matrix3<Scalar> R;
        R = Eigen::AngleAxis<Scalar>(theta, GraphOptimizor::Vector3<Scalar>::UnitZ());
        GraphOptimizor::Vector3<Scalar> t = GraphOptimizor::Vector3<Scalar>(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        cameraPoses.push_back(Frame(R, t));
    }

    // 随机数生成三维特征点
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
    for (int j = 0; j < featureNums; ++j) {
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(4., 8.);

        GraphOptimizor::Vector3<Scalar> Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        points.push_back(Pw);

        // 在每一帧上的观测量
        for (int i = 0; i < poseNums; ++i) {
            GraphOptimizor::Vector3<Scalar> Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            if (Pc.z() < 1e-10) {
                continue;
            }
            Pc = Pc / Pc.z();  // 归一化图像平面
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);
            cameraPoses[i].featurePerId.insert(std::make_pair(j, Pc));
        }
    }
}





/* 程序主函数入口 */
int main() {
    std::cout << "Test GraphOptimizor solver on Mono BA problem." << std::endl;

    // 第一步：准备测试数据
    std::cout << "\nStep 1: Prepare dataset." << std::endl;
    std::vector<Frame> cameras;
    std::vector<GraphOptimizor::Vector3<Scalar>> points;
    GetSimDataInWordFrame(cameras, points);

    // 第二步：构造待求解问题
    std::cout << "\nStep 2: Construct GraphOptimizor solver." << std::endl;
    GraphOptimizor::Problem<Scalar> problem;
    size_t type_cameraVertex = 0;
    size_t type_landmarkVertex = 1;

    // 第三步：构造相机 pose 节点，并添加到 problem 中
    std::cout << "\nStep 3: Add camera pose vertices." << std::endl;
    std::default_random_engine generator;
    std::normal_distribution<double> camera_rotation_noise(0., 0.05);
    std::normal_distribution<double> camera_position_noise(0., 0.1);
    std::vector<std::shared_ptr<GraphOptimizor::VertexPose<Scalar>>> cameraVertices;
    for (size_t i = 0; i < cameras.size(); ++i) {
        // 给相机位置和姿态初值增加噪声
        GraphOptimizor::Matrix3<Scalar> noise_R, noise_X, noise_Y, noise_Z;
        noise_X = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), GraphOptimizor::Vector3<Scalar>::UnitX());
        noise_Y = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), GraphOptimizor::Vector3<Scalar>::UnitY());
        noise_Z = Eigen::AngleAxis<Scalar>(camera_rotation_noise(generator), GraphOptimizor::Vector3<Scalar>::UnitZ());
        noise_R = noise_X * noise_Y * noise_Z;
        GraphOptimizor::Vector3<Scalar> noise_t(camera_position_noise(generator), camera_position_noise(generator), camera_position_noise(generator));
        GraphOptimizor::Quaternion<Scalar> noise_q_wc(noise_R);
        if (i < 2) {
            noise_t.setZero();
            noise_q_wc.setIdentity();
        }
        GraphOptimizor::Quaternion<Scalar> temp_q_wc = cameras[i].qwc * noise_q_wc;
        GraphOptimizor::Vector3<Scalar> temp_t_wc = cameras[i].twc + noise_t;

        GraphOptimizor::VectorX<Scalar> param(7);
        param << temp_t_wc, temp_q_wc.x(), temp_q_wc.y(), temp_q_wc.z(), temp_q_wc.w();
        std::shared_ptr<GraphOptimizor::VertexPose<Scalar>> cameraVertex(new GraphOptimizor::VertexPose<Scalar>());
        cameraVertex->SetParameters(param);
        cameraVertex->SetType(type_cameraVertex);
        cameraVertices.emplace_back(cameraVertex);
        problem.AddVertex(cameraVertex);
    }
    std::cout << "Add " << cameraVertices.size() << " camera vertices, problem has " << problem.GetVertexNum() << " vertices." << std::endl;

    // 第四步：构造特征点 position 节点，同时构造 reprojection 边，添加到 problem 中
    std::cout << "\nStep 4: Add landmark position vertices." << std::endl;
    std::vector<std::shared_ptr<GraphOptimizor::VertexLandmarkPosition<Scalar>>> landmarkVertices;
    std::normal_distribution<double> landmark_position_noise(0.0, 0.5);
    for (size_t i = 0; i < points.size(); ++i) {
        // 构造特征点的节点
        GraphOptimizor::VectorX<Scalar> param = points[i] + GraphOptimizor::Vector3<Scalar>(
            landmark_position_noise(generator), landmark_position_noise(generator), landmark_position_noise(generator));
        std::shared_ptr<GraphOptimizor::VertexLandmarkPosition<Scalar>> landmarkVertex(new GraphOptimizor::VertexLandmarkPosition<Scalar>());
        landmarkVertex->SetParameters(param);
        landmarkVertex->SetType(type_landmarkVertex);
        landmarkVertices.emplace_back(landmarkVertex);
        problem.AddVertex(landmarkVertex);

        // 遍历此点的观测，构造 reprojection 边，添加到 problem 中
        for (size_t j = 0; j < cameras.size(); ++j) {
            GraphOptimizor::Vector2<Scalar> normPoint = cameras[j].featurePerId.find(i)->second.head<2>();
            std::shared_ptr<GraphOptimizor::EdgeReprojectionPos<Scalar>> edge(new GraphOptimizor::EdgeReprojectionPos<Scalar>(normPoint));
            edge->AddVertex(landmarkVertex, 0);
            edge->AddVertex(cameraVertices[j], 1);
            std::shared_ptr<GraphOptimizor::HuberKernel<Scalar>> kernel(new GraphOptimizor::HuberKernel<Scalar>(1.0));
            edge->SetKernel(kernel);
            problem.AddEdge(edge);
        }
    }
    std::cout << "Add " << landmarkVertices.size() << " landmark vertices, problem has " << problem.GetVertexNum() << " vertices and " <<
        problem.GetEdgeNum() << " edges." << std::endl;
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        std::cout << "Camera " << i << " has " << problem.GetRelativeEdges(cameraVertices[i]).size() << " relative edges." << std::endl;
    }

    // 第五步：删除操作
    auto deletedVertex = landmarkVertices[0];
    problem.RemoveVertex(deletedVertex);
    std::cout << "After delete" << std::endl;
    std::cout << "problem has " << problem.GetVertexNum() << " vertices and " << problem.GetEdgeNum() << " edges." << std::endl;
    for (size_t i = 0; i < cameraVertices.size(); ++i) {
        auto relativeEdges = problem.GetRelativeEdges(cameraVertices[i]);
        std::cout << "Camera " << i << " has " << relativeEdges.size() << " relative edges." << std::endl;
        for (auto edge : relativeEdges) {
            auto vertices = edge->GetVertices();
            for (auto v : vertices) {
                // 如果找到了被删除的节点，那说明有问题
                if (v == deletedVertex) {
                    std::cout << "check error!" << std::endl;
                }
            }
        }
    }


    return 0;
}