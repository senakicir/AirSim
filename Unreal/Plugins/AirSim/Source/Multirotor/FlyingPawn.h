#pragma once

#include "VehiclePawnWrapper.h"
#include "GameFramework/RotatingMovementComponent.h"
#include <memory>
#include "vehicles/multirotor/controllers/DroneCommon.hpp"
#include "PIPCamera.h"
#include "DroneInterface.h"
#include "FlyingPawn.generated.h"

UCLASS()
class AIRSIM_API AFlyingPawn : public APawn, public IDroneInterface
{
    GENERATED_BODY()

public:
    AFlyingPawn();
    //sena was here
    FVector droneLocationUpdated;
    FRotator droneOrientationUpdated;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debugging")
    float RotatorFactor = 1.0f;

    void setRotorSpeed(int rotor_index, float radsPerSec);
    void initializeForBeginPlay();
    VehiclePawnWrapper* getVehiclePawnWrapper();

    virtual void NotifyHit(class UPrimitiveComponent* MyComp, class AActor* Other, class UPrimitiveComponent* OtherComp, bool bSelfMoved, FVector HitLocation,
        FVector HitNormal, FVector NormalImpulse, const FHitResult& Hit) override;
   
    virtual void Tick(float DeltaTime) override; //sena was here
    //sena was here
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "MyCategory")
    FVector getDronePositionUpdated();
    virtual FVector getDronePositionUpdated_Implementation() override;
    
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "MyCategory")
    FRotator getDroneOrientationUpdated();
    virtual FRotator getDroneOrientationUpdated_Implementation() override;

private: //methods
    void setupComponentReferences();

private: //variables
         //Unreal components
    static constexpr size_t rotor_count = 4;
    UPROPERTY() APIPCamera* fpv_camera_front_left_;
    UPROPERTY() APIPCamera* fpv_camera_front_right_;
    UPROPERTY() APIPCamera* fpv_camera_front_center_;
    UPROPERTY() APIPCamera* fpv_camera_back_center_;
    UPROPERTY() APIPCamera* fpv_camera_bottom_center_;

    UPROPERTY() URotatingMovementComponent* rotating_movements_[rotor_count];

    std::unique_ptr<VehiclePawnWrapper> wrapper_;

};
