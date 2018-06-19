// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef air_RpcLibAdapatorsBase_hpp
#define air_RpcLibAdapatorsBase_hpp

#include "common/Common.hpp"
#include "common/CommonStructs.hpp"
#include "physics/Kinematics.hpp"
#include "physics/Environment.hpp"
#include "common/ImageCaptureBase.hpp"
#include "safety/SafetyEval.hpp"
#include "rpc/msgpack.hpp"


namespace msr { namespace airlib_rpclib {

class RpcLibAdapatorsBase {
public:
    template<typename TSrc, typename TDest>
    static void to(const std::vector<TSrc>& s, std::vector<TDest>& d)
    {
        d.clear();
        for (size_t i = 0; i < s.size(); ++i)
            d.push_back(s.at(i).to());
    }

    template<typename TSrc, typename TDest>
    static void from(const std::vector<TSrc>& s, std::vector<TDest>& d)
    {
        d.clear();
        for (size_t i = 0; i < s.size(); ++i)
            d.push_back(TDest(s.at(i)));
    }

    struct Vector3r {
        msr::airlib::real_T x_val = 0, y_val = 0, z_val = 0;
        MSGPACK_DEFINE_MAP(x_val, y_val, z_val);

        Vector3r()
        {}

        Vector3r(const msr::airlib::Vector3r& s)
        {
            x_val = s.x();
            y_val = s.y();
            z_val = s.z();
        }
        msr::airlib::Vector3r to() const
        {
            return msr::airlib::Vector3r(x_val, y_val, z_val);
        }
    };

    //sena was here
    struct Vector3r_arr {
        Vector3r dronePos;
        Vector3r droneOrient;
        Vector3r humanPos;
        Vector3r hip;
        Vector3r right_up_leg;
        Vector3r right_leg;
        Vector3r right_foot;
        Vector3r left_up_leg;
        Vector3r left_leg;
        Vector3r left_foot;
        Vector3r spine1;
        Vector3r neck;
        Vector3r head;
        Vector3r head_top;
        Vector3r left_arm;
        Vector3r left_forearm;
        Vector3r left_hand;
        Vector3r right_arm;
        Vector3r right_forearm;
        Vector3r right_hand;
        
        Vector3r right_hand_tip;
        Vector3r left_hand_tip;
        Vector3r right_foot_tip;
        Vector3r left_foot_tip;
        MSGPACK_DEFINE_MAP(dronePos, droneOrient, humanPos, hip,
                           right_up_leg, right_leg, right_foot,
                           left_up_leg,left_leg, left_foot,
                           spine1,neck,head,head_top,
                           left_arm,left_forearm,left_hand,
                           right_arm,right_forearm,right_hand,
                           right_hand_tip, left_hand_tip, right_foot_tip, left_foot_tip);
        
        Vector3r_arr()
        {}
        
        Vector3r_arr(const msr::airlib::Vector3r_arr& s)
        {
            dronePos = s.dronePos;
            droneOrient = s.droneOrient;
            humanPos = s.humanPos;
            hip = s.hip;
            right_up_leg = s.right_up_leg;
            right_leg = s.right_leg;
            right_foot = s.right_foot;
            left_up_leg = s.left_up_leg;
            left_leg = s.left_leg;
            left_foot = s.left_foot;
            spine1 = s.spine1;
            neck = s.neck;
            head = s.head;
            head_top = s.head_top;
            left_arm = s.left_arm;
            left_forearm = s.left_forearm;
            left_hand = s.left_hand;
            right_arm = s.right_arm;
            right_forearm = s.right_forearm;
            right_hand = s.right_hand;
            right_hand_tip = s.right_hand_tip;
            left_hand_tip = s.left_hand_tip;
            right_foot_tip = s.right_foot_tip;
            left_foot_tip = s.left_foot_tip;
        }
        
        msr::airlib::Vector3r_arr to() const
        {
            return msr::airlib::Vector3r_arr(dronePos.to(), droneOrient.to(), humanPos.to(), hip.to(),right_up_leg.to(), right_leg.to(), right_foot.to(), left_up_leg.to(),left_leg.to(), left_foot.to(), spine1.to(),neck.to(),head.to(),head_top.to(), left_arm.to(),left_forearm.to(),left_hand.to(),right_arm.to(),right_forearm.to(),right_hand.to(), right_hand_tip.to(), left_hand_tip.to(), right_foot_tip.to(), left_foot_tip.to());
        }
    };
    
    struct CollisionInfo {
        bool has_collided = false;
        Vector3r normal;
        Vector3r impact_point;
        Vector3r position;
        msr::airlib::real_T penetration_depth = 0;
        msr::airlib::TTimePoint time_stamp = 0;
        std::string object_name;
        int object_id = -1;

        MSGPACK_DEFINE_MAP(has_collided, penetration_depth, time_stamp, normal, impact_point, position, object_name, object_id);
        
        CollisionInfo()
        {}

        CollisionInfo(const msr::airlib::CollisionInfo& s)
        {
            has_collided = s.has_collided;
            normal = s.normal;
            impact_point = s.impact_point;
            position = s.position;
            penetration_depth = s.penetration_depth;
            time_stamp = s.time_stamp;
            object_name = s.object_name;
            object_id = s.object_id;
        }

        msr::airlib::CollisionInfo to() const
        {
            return msr::airlib::CollisionInfo(has_collided, normal.to(), impact_point.to(), position.to(),
                penetration_depth, time_stamp, object_name, object_id);
        }
    };

    struct Quaternionr {
        msr::airlib::real_T w_val = 1, x_val = 0, y_val = 0, z_val = 0;
        MSGPACK_DEFINE_MAP(w_val, x_val, y_val, z_val);

        Quaternionr()
        {}

        Quaternionr(const msr::airlib::Quaternionr& s)
        {
            w_val = s.w();
            x_val = s.x();
            y_val = s.y();
            z_val = s.z();
        }
        msr::airlib::Quaternionr to() const
        {
            return msr::airlib::Quaternionr(w_val, x_val, y_val, z_val);
        }
    };

    struct Pose {
        Vector3r position;
        Quaternionr orientation;
        MSGPACK_DEFINE_MAP(position, orientation);

        Pose()
        {}
        Pose(const msr::airlib::Pose& s)
        {
            position = s.position;
            orientation = s.orientation;
        }
        msr::airlib::Pose to() const
        {
            return msr::airlib::Pose(position.to(), orientation.to());
        }
    };

    struct GeoPoint {
        double latitude = 0, longitude = 0;
        float altitude = 0;
        MSGPACK_DEFINE_MAP(latitude, longitude, altitude);

        GeoPoint()
        {}

        GeoPoint(const msr::airlib::GeoPoint& s)
        {
            latitude = s.latitude;
            longitude = s.longitude;
            altitude = s.altitude;
        }
        msr::airlib::GeoPoint to() const
        {
            return msr::airlib::GeoPoint(latitude, longitude, altitude);
        }
    };

    struct RCData {
        uint64_t timestamp = 0;
        float pitch = 0, roll = 0, throttle = 0, yaw = 0;
        float left_z = 0, right_z = 0;
        uint16_t switches = 0;
        std::string vendor_id = "";
        bool is_initialized = false; //is RC connected?
        bool is_valid = false; //must be true for data to be valid

        MSGPACK_DEFINE_MAP(timestamp, pitch, roll, throttle, yaw, left_z, right_z, switches, vendor_id, is_initialized, is_valid);

        RCData()
        {}

        RCData(const msr::airlib::RCData& s)
        {
            timestamp = s.timestamp;
            pitch = s.pitch;
            roll = s.roll;
            throttle = s.throttle;
            yaw = s.yaw;
            left_z = s.left_z;
            right_z = s.right_z;
            switches = s.switches;
            vendor_id = s.vendor_id;
            is_initialized = s.is_initialized;
            is_valid = s.is_valid;

        }
        msr::airlib::RCData to() const
        {
            msr::airlib::RCData d;
            d.timestamp = timestamp;
            d.pitch = pitch;
            d.roll = roll;
            d.throttle = throttle;
            d.yaw = yaw;
            d.left_z = left_z;
            d.right_z = right_z;
            d.switches = switches;
            d.vendor_id = vendor_id;
            d.is_initialized = is_initialized;
            d.is_valid = is_valid;
            
            return d;
        }
    };

    struct CameraInfo {
        Pose pose;
        float fov;

        MSGPACK_DEFINE_MAP(pose, fov);

        CameraInfo()
        {}

        CameraInfo(const msr::airlib::CameraInfo& s)
        {
            pose = s.pose;
            fov = s.fov;
        }

        msr::airlib::CameraInfo to() const
        {
            msr::airlib::CameraInfo s;
            s.pose = pose.to();
            s.fov = fov;

            return s;
        }
    };
    
    struct KinematicsState {
        Vector3r position;
        Quaternionr orientation;

        Vector3r linear_velocity;
        Vector3r angular_velocity;

        Vector3r linear_acceleration;
        Vector3r angular_acceleration;

        MSGPACK_DEFINE_MAP(position, orientation, linear_velocity, angular_velocity, linear_acceleration, angular_acceleration);


        KinematicsState()
        {}

        KinematicsState(const msr::airlib::Kinematics::State& s)
        {
            position = s.pose.position;
            orientation = s.pose.orientation;
            linear_velocity = s.twist.linear;
            angular_velocity = s.twist.angular;
            linear_acceleration = s.accelerations.linear;
            angular_acceleration = s.accelerations.angular;
        }

        msr::airlib::Kinematics::State to() const
        {
            msr::airlib::Kinematics::State s;
            s.pose.position = position.to();
            s.pose.orientation = orientation.to();
            s.twist.linear = linear_velocity.to();
            s.twist.angular = angular_velocity.to();
            s.accelerations.linear = linear_acceleration.to();
            s.accelerations.angular = angular_acceleration.to();

            return s;
        }
    };

    struct EnvironmentState {
        Vector3r position;
        GeoPoint geo_point;

        //these fields are computed
        Vector3r gravity;
        float air_pressure;
        float temperature;
        float air_density;

        MSGPACK_DEFINE_MAP(position, geo_point, gravity, air_pressure, temperature, air_density);

        EnvironmentState()
        {}

        EnvironmentState(const msr::airlib::Environment::State& s)
        {
            position = s.position;
            geo_point = s.geo_point;
            gravity = s.gravity;
            air_pressure = s.air_pressure;
            temperature = s.temperature;
            air_density = s.air_density;
        }

        msr::airlib::Environment::State to() const
        {
            msr::airlib::Environment::State s;
            s.position = position.to();
            s.geo_point = geo_point.to();
            s.gravity = gravity.to();
            s.air_pressure = air_pressure;
            s.temperature = temperature;
            s.air_density = air_density;

            return s;
        }
    };

    struct ImageRequest {
        std::string camera_name;
        msr::airlib::ImageCaptureBase::ImageType image_type;
        bool pixels_as_float;
        bool compress;

        MSGPACK_DEFINE_MAP(camera_name, image_type, pixels_as_float, compress);

        ImageRequest()
        {}

        ImageRequest(const msr::airlib::ImageCaptureBase::ImageRequest& s)
        {
            camera_name = s.camera_name;
            image_type = s.image_type;
            pixels_as_float = s.pixels_as_float;
            compress = s.compress;
        }

        msr::airlib::ImageCaptureBase::ImageRequest to() const
        {
            msr::airlib::ImageCaptureBase::ImageRequest d;
            d.camera_name = camera_name;
            d.image_type = image_type;
            d.pixels_as_float = pixels_as_float;
            d.compress = compress;

            return d;
        }

        static std::vector<ImageRequest> from(
            const std::vector<msr::airlib::ImageCaptureBase::ImageRequest>& request
        ) {
            std::vector<ImageRequest> request_adaptor;
            for (const auto& item : request)
                request_adaptor.push_back(ImageRequest(item));

            return request_adaptor;
        }
        static std::vector<msr::airlib::ImageCaptureBase::ImageRequest> to(
            const std::vector<ImageRequest>& request_adapter
        ) {
            std::vector<msr::airlib::ImageCaptureBase::ImageRequest> request;
            for (const auto& item : request_adapter)
                request.push_back(item.to());

            return request;
        }         
    };

    struct ImageResponse {
        std::vector<uint8_t> image_data_uint8;
        std::vector<float> image_data_float;

        Vector3r camera_position;
        Quaternionr camera_orientation;
        //sena was here
        Vector3r_arr bones;
        msr::airlib::TTimePoint time_stamp;
        std::string message;
        bool pixels_as_float;
        bool compress;
        int width, height;
        msr::airlib::ImageCaptureBase::ImageType image_type;

        MSGPACK_DEFINE_MAP(image_data_uint8, image_data_float, camera_position, 
            camera_orientation, bones, time_stamp, message, pixels_as_float, compress, width, height, image_type);

        ImageResponse()
        {}

        ImageResponse(const msr::airlib::ImageCaptureBase::ImageResponse& s)
        {
            pixels_as_float = s.pixels_as_float;
            
            image_data_uint8 = s.image_data_uint8;
            image_data_float = s.image_data_float;

            //TODO: remove bug workaround for https://github.com/rpclib/rpclib/issues/152
            if (image_data_uint8.size() == 0)
                image_data_uint8.push_back(0);
            if (image_data_float.size() == 0)
                image_data_float.push_back(0);

            camera_position = Vector3r(s.camera_position);
            camera_orientation = Quaternionr(s.camera_orientation);
            bones = s.bones;
            time_stamp = s.time_stamp;
            message = s.message;
            compress = s.compress;
            width = s.width;
            height = s.height;
            image_type = s.image_type;
        }

        msr::airlib::ImageCaptureBase::ImageResponse to() const
        {
            msr::airlib::ImageCaptureBase::ImageResponse d;

            d.pixels_as_float = pixels_as_float;

            if (! pixels_as_float)
                d.image_data_uint8 = image_data_uint8;
            else
                d.image_data_float = image_data_float;

            d.camera_position = camera_position.to();
            d.camera_orientation = camera_orientation.to();
            d.bones = bones.to();
            d.time_stamp = time_stamp;
            d.message = message;
            d.compress = compress;
            d.width = width;
            d.height = height;
            d.image_type = image_type;

            return d;
        }

        static std::vector<msr::airlib::ImageCaptureBase::ImageResponse> to(
            const std::vector<ImageResponse>& response_adapter
        ) {
            std::vector<msr::airlib::ImageCaptureBase::ImageResponse> response;
            for (const auto& item : response_adapter)
                response.push_back(item.to());

            return response;
        }
        static std::vector<ImageResponse> from(
            const std::vector<msr::airlib::ImageCaptureBase::ImageResponse>& response
        ) {
            std::vector<ImageResponse> response_adapter;
            for (const auto& item : response)
                response_adapter.push_back(ImageResponse(item));

            return response_adapter;
        }
    };
};

}} //namespace

MSGPACK_ADD_ENUM(msr::airlib::SafetyEval::SafetyViolationType_);
MSGPACK_ADD_ENUM(msr::airlib::SafetyEval::ObsAvoidanceStrategy);
MSGPACK_ADD_ENUM(msr::airlib::ImageCaptureBase::ImageType);


#endif
