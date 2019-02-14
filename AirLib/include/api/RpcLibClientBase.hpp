// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef air_RpcLibClientBase_hpp
#define air_RpcLibClientBase_hpp

#include "common/Common.hpp"
#include "common/CommonStructs.hpp"
#include "common/ImageCaptureBase.hpp"
#include "physics/Kinematics.hpp"
#include "physics/Environment.hpp"

namespace msr { namespace airlib {

//common methods for RCP clients of different vehicles
class RpcLibClientBase {
public:
    enum class ConnectionState : uint {
        Initial = 0, Connected, Disconnected, Reset, Unknown
    };
public:
    RpcLibClientBase(const string& ip_address = "localhost", uint16_t port = 41451, float timeout_sec = 60);
    virtual ~RpcLibClientBase();    //required for pimpl

    void confirmConnection();
    void reset();

    ConnectionState getConnectionState();
    bool ping();
    int getClientVersion() const;
    int getServerVersion() const;
    int getMinRequiredServerVersion() const;
    int getMinRequiredClientVersion() const;

    bool simIsPaused() const;
    void simPause(bool is_paused);
    void simPauseHuman(bool is_paused); //sena was here
    void simPauseDrone(bool is_paused); //sena was here
    
    void simContinueForTime(double seconds);

    Pose simGetObjectPose(const std::string& object_name) const;

    //task management APIs
    void cancelLastTask(const std::string& vehicle_name = "");
    virtual RpcLibClientBase* waitOnLastTask(bool* task_result = nullptr, float timeout_sec = Utils::nan<float>());

    bool simSetSegmentationObjectID(const std::string& mesh_name, int object_id, bool is_name_regex = false);
    int simGetSegmentationObjectID(const std::string& mesh_name) const;
    void simPrintLogMessage(const std::string& message, std::string message_param = "", unsigned char severity = 0);


    bool armDisarm(bool arm, const std::string& vehicle_name = "");
    bool isApiControlEnabled(const std::string& vehicle_name = "") const;
    void enableApiControl(bool is_enabled, const std::string& vehicle_name = "");

    msr::airlib::GeoPoint getHomeGeoPoint(const std::string& vehicle_name = "") const;
    Pose simGetVehiclePose(const std::string& vehicle_name = "") const;
    void simSetVehiclePose(const Pose& pose, bool ignore_collision, const std::string& vehicle_name = "");
    void simSetVehiclePose_senaver(const Pose& pose, const std::string& vehicle_name = ""); //sena was here

    vector<ImageCaptureBase::ImageResponse> simGetImages(vector<ImageCaptureBase::ImageRequest> request, const std::string& vehicle_name = "");
    vector<uint8_t> simGetImage(const std::string& camera_name, ImageCaptureBase::ImageType type, const std::string& vehicle_name = "");

    CollisionInfo simGetCollisionInfo(const std::string& vehicle_name = "") const;

    //sena was here
    Vector3r_arr getBonePositions(const std::string& vehicle_name = "") const;
    Vector3r getInitialDronePos(const std::string& vehicle_name = "") const;
    void changeAnimation(int new_anim, const std::string& vehicle_name = "") const;
    void changeCalibrationMode(bool calib_mode, const std::string& vehicle_name = "") const;

    CameraInfo simGetCameraInfo(const std::string& camera_name, const std::string& vehicle_name = "") const;
    void simSetCameraOrientation(const std::string& camera_name, const Quaternionr& orientation, const std::string& vehicle_name = "");

    msr::airlib::Kinematics::State simGetGroundTruthKinematics(const std::string& vehicle_name = "") const;
    msr::airlib::Environment::State simGetGroundTruthEnvironment(const std::string& vehicle_name = "") const;

protected:
    void* getClient();
    const void* getClient() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
};

}} //namespace
#endif
