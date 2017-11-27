#pragma once

#include "CoreMinimal.h"
#include "PIPCamera.h"
#include "controllers/VehicleCameraBase.hpp"
#include "common/CommonStructs.hpp"
#include "common/CommonStructs.hpp"

class VehicleCameraConnector : public msr::airlib::VehicleCameraBase
{
public:
    typedef msr::airlib::VehicleCameraBase::ImageType ImageType;
    typedef msr::airlib::Vector3r_arr Vector3r_arr;

    VehicleCameraConnector(APIPCamera* camera);
    virtual ~VehicleCameraConnector();
    //sena was here
    virtual ImageResponse getImage(ImageType image_type, bool pixels_as_float, bool compress, Vector3r_arr* bonePos) override;
private:
    //sena was here
    msr::airlib::VehicleCameraBase::ImageResponse getSceneCaptureImage(ImageType image_type, bool pixels_as_float, bool compress, bool use_safe_method, Vector3r_arr* bonePos);
    
    void addScreenCaptureHandler(UWorld *world);
    bool getScreenshotScreen(ImageType image_type, std::vector<uint8_t>& compressedPng);
    
private:
    APIPCamera* camera_;
    std::vector<uint8_t> last_compressed_png_;
};

