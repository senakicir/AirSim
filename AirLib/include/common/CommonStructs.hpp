// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef msr_airlib_CommonStructs_hpp
#define msr_airlib_CommonStructs_hpp

#include "common/Common.hpp"
#include <ostream>

namespace msr { namespace airlib {

//velocity
struct Twist {
    Vector3r linear, angular;

    Twist()
    {}

    Twist(const Vector3r& linear_val, const Vector3r& angular_val)
        : linear(linear_val), angular(angular_val)
    {
    }

    static const Twist zero()
    {
        static const Twist zero_twist(Vector3r::Zero(), Vector3r::Zero());
        return zero_twist;
    }
};

//force & torque
struct Wrench {
    Vector3r force, torque;

    Wrench()
    {}

    Wrench(const Vector3r& force_val, const Vector3r& torque_val)
        : force(force_val), torque(torque_val)
    {
    }

    //support basic arithmatic 
    Wrench operator+(const Wrench& other) const
    {
        Wrench result;
        result.force = this->force + other.force;
        result.torque = this->torque + other.torque;
        return result;
    }
    Wrench operator+=(const Wrench& other)
    {
        force += other.force;
        torque += other.torque;
        return *this;
    }
    Wrench operator-(const Wrench& other) const
    {
        Wrench result;
        result.force = this->force - other.force;
        result.torque = this->torque - other.torque;
        return result;
    }
    Wrench operator-=(const Wrench& other)
    {
        force -= other.force;
        torque -= other.torque;
        return *this;
    }

    static const Wrench zero()
    {
        static const Wrench zero_wrench(Vector3r::Zero(), Vector3r::Zero());
        return zero_wrench;
    }
};

struct Momentums {
    Vector3r linear;
    Vector3r angular;

    Momentums()
    {}

    Momentums(const Vector3r& linear_val, const Vector3r& angular_val)
        : linear(linear_val), angular(angular_val)
    {
    }

    static const Momentums zero()
    {
        static const Momentums zero_val(Vector3r::Zero(), Vector3r::Zero());
        return zero_val;
    }
};

struct Accelerations {
    Vector3r linear;
    Vector3r angular;

    Accelerations()
    {}

    Accelerations(const Vector3r& linear_val, const Vector3r& angular_val)
        : linear(linear_val), angular(angular_val)
    {
    }

    static const Accelerations zero()
    {
        static const Accelerations zero_val(Vector3r::Zero(), Vector3r::Zero());
        return zero_val;
    }
};

struct PoseWithCovariance {
    VectorMath::Pose pose;
    vector<real_T> covariance;	//36 elements, 6x6 matrix

    PoseWithCovariance()
        : covariance(36, 0)
    {}
};

struct PowerSupply {
    vector<real_T> voltage, current;
};

struct TwistWithCovariance {
    Twist twist;
    vector<real_T> covariance;	//36 elements, 6x6 matrix

    TwistWithCovariance()
        : covariance(36, 0)
    {}
};

struct Joystick {
    vector<float> axes;
    vector<int> buttons;
};

struct Odometry {
    PoseWithCovariance pose;
    TwistWithCovariance twist;
};

struct GeoPoint {
    double latitude = 0, longitude = 0;
    float altitude = 0;

    GeoPoint()
    {}

    GeoPoint(double latitude_val, double longitude_val, float altitude_val)
    {
        set(latitude_val, longitude_val, altitude_val);
    }

    void set(double latitude_val, double longitude_val, float altitude_val)
    {
        latitude = latitude_val, longitude = longitude_val; altitude = altitude_val;
    }

    friend std::ostream& operator<<(std::ostream &os, GeoPoint const &g) { 
        return os << "[" << g.latitude << ", " << g.longitude << ", " << g.altitude << "]";
    }

    std::string to_string()
    {
        return std::to_string(latitude) + string(", ") + std::to_string(longitude) + string(", ") + std::to_string(altitude);
    }
};

struct HomeGeoPoint {
    GeoPoint home_point;
    double lat_rad, lon_rad;
    double cos_lat, sin_lat;

    HomeGeoPoint()
    {}
    HomeGeoPoint(const GeoPoint& home_point_val)
    {
        initialize(home_point_val);
    }
    void initialize(const GeoPoint& home_point_val)
    {
        home_point = home_point_val;
        lat_rad = Utils::degreesToRadians(home_point.latitude);
        lon_rad = Utils::degreesToRadians(home_point.longitude);
        cos_lat = cos(lat_rad);
        sin_lat = sin(lat_rad);
    }
};

//sena was here
struct Vector3r_arr {
    Vector3r dronePos = Vector3r::Zero();
    Vector3r droneOrient = Vector3r::Zero();
    Vector3r humanPos = Vector3r::Zero();
    Vector3r hip = Vector3r::Zero();
    Vector3r right_up_leg = Vector3r::Zero();
    Vector3r right_leg = Vector3r::Zero();
    Vector3r right_foot = Vector3r::Zero();
    Vector3r left_up_leg = Vector3r::Zero();
    Vector3r left_leg = Vector3r::Zero();
    Vector3r left_foot = Vector3r::Zero();
    Vector3r spine1 = Vector3r::Zero();
    Vector3r neck = Vector3r::Zero();
    Vector3r head = Vector3r::Zero();
    Vector3r head_top = Vector3r::Zero();
    Vector3r left_arm = Vector3r::Zero();
    Vector3r left_forearm = Vector3r::Zero();
    Vector3r left_hand = Vector3r::Zero();
    Vector3r right_arm = Vector3r::Zero();
    Vector3r right_forearm = Vector3r::Zero();
    Vector3r right_hand = Vector3r::Zero();
    
    Vector3r right_hand_tip = Vector3r::Zero();
    Vector3r left_hand_tip = Vector3r::Zero();
    Vector3r right_foot_tip = Vector3r::Zero();
    Vector3r left_foot_tip = Vector3r::Zero();
    
    Vector3r_arr()
    {}
    
    Vector3r_arr(const Vector3r& dronePos_val, const Vector3r& droneOrient_val, const Vector3r& humanPos_val, const Vector3r& hip_val,
                 const Vector3r& right_up_leg_val, const Vector3r& right_leg_val, const Vector3r& right_foot_val,
                 const Vector3r& left_up_leg_val, const Vector3r& left_leg_val, const Vector3r& left_foot_val,
                 const Vector3r& spine1_val, const Vector3r& neck_val, const Vector3r& head_val, const Vector3r& head_top_val,
                 const Vector3r& left_arm_val, const Vector3r& left_forearm_val, const Vector3r& left_hand_val,
                 const Vector3r& right_arm_val, const Vector3r& right_forearm_val, const Vector3r& right_hand_val,
                 const Vector3r& right_hand_tip_val, const Vector3r& left_hand_tip_val, const Vector3r& right_foot_tip_val, const Vector3r& left_foot_tip_val):
    dronePos(dronePos_val), droneOrient(droneOrient_val), humanPos(humanPos_val), hip(hip_val),
    right_up_leg(right_up_leg_val), right_leg(right_leg_val), right_foot(right_foot_val),
    left_up_leg(left_up_leg_val), left_leg(left_leg_val), left_foot(left_foot_val),
    spine1(spine1_val), neck(neck_val), head(head_val), head_top(head_top_val),
    left_arm(left_arm_val) , left_forearm(left_forearm_val), left_hand(left_hand_val),
    right_arm(right_arm_val), right_forearm(right_forearm_val), right_hand(right_hand_val),
    right_hand_tip(right_hand_tip_val), left_hand_tip(left_hand_tip_val), right_foot_tip(right_foot_tip_val), left_foot_tip(left_foot_tip_val)
    {
    }
};

struct CollisionInfo {
    bool has_collided = false;
    Vector3r normal = Vector3r::Zero();
    Vector3r impact_point = Vector3r::Zero();
    Vector3r position = Vector3r::Zero();
    real_T penetration_depth = 0;
    TTimePoint time_stamp = 0;
    unsigned int collision_count = 0;
    std::string object_name;
    int object_id = -1;

    CollisionInfo()
    {}

    CollisionInfo(bool has_collided_val, const Vector3r& normal_val, 
        const Vector3r& impact_point_val, const Vector3r& position_val, 
        real_T penetration_depth_val, TTimePoint time_stamp_val,
        const std::string& object_name_val, int object_id_val)
        : has_collided(has_collided_val), normal(normal_val),
        impact_point(impact_point_val), position(position_val),
        penetration_depth(penetration_depth_val), time_stamp(time_stamp_val),
        object_name(object_name_val), object_id(object_id_val)
    {
    }
};

struct CameraInfo {
    Pose pose;
    float fov;

    CameraInfo()
    {}

    CameraInfo(const Pose& pose_val, float fov_val)
        : pose(pose_val), fov(fov_val)
    {
    }
};

struct CollisionResponseInfo {
    unsigned int collision_count_raw = 0;
    unsigned int collision_count_non_resting = 0;
    TTimePoint collision_time_stamp = 0;
};

struct GeoPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Quaternionr orientation;
    GeoPoint position;
};



}} //namespace
#endif
