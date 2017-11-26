#include "SimModeWorldMultiRotor.h"
#include "ConstructorHelpers.h"
#include "AirBlueprintLib.h"
#include "vehicles/multirotor/controllers/DroneControllerBase.hpp"
#include "physics/PhysicsBody.hpp"
#include <memory>
#include "Logging/MessageLog.h"
#include "vehicles/multirotor/MultiRotorParamsFactory.hpp"


ASimModeWorldMultiRotor::ASimModeWorldMultiRotor()
{
    static ConstructorHelpers::FClassFinder<APIPCamera> external_camera_class(TEXT("Blueprint'/AirSim/Blueprints/BP_PIPCamera'"));
    external_camera_class_ = external_camera_class.Succeeded() ? external_camera_class.Class : nullptr;
    static ConstructorHelpers::FClassFinder<ACameraDirector> camera_director_class(TEXT("Blueprint'/AirSim/Blueprints/BP_CameraDirector'"));
    camera_director_class_ = camera_director_class.Succeeded() ? camera_director_class.Class : nullptr;
    static ConstructorHelpers::FClassFinder<TMultiRotorPawn> vehicle_pawn_class(TEXT("Blueprint'/AirSim/Blueprints/BP_FlyingPawn'"));
    vehicle_pawn_class_ = vehicle_pawn_class.Succeeded() ? vehicle_pawn_class.Class : nullptr;
}

void ASimModeWorldMultiRotor::BeginPlay()
{
    Super::BeginPlay();
    
    if (fpv_vehicle_connector_ != nullptr) {
        //create its control server
        try {
            fpv_vehicle_connector_->startApiServer();
        }
        catch (std::exception& ex) {
            UAirBlueprintLib::LogMessageString("Cannot start RpcLib Server", ex.what(), LogDebugLevel::Failure);
        }
    }
    
}

void ASimModeWorldMultiRotor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    //stop physics thread before we dismental
    stopAsyncUpdator();
    
    if (fpv_vehicle_connector_ != nullptr) {
        fpv_vehicle_connector_->stopApiServer();
        fpv_vehicle_pawn_wrapper_ = nullptr;
    }
    
    //for (AActor* actor : spawned_actors_) {
    //    actor->Destroy();
    //}
    spawned_actors_.Empty();
    if (CameraDirector != nullptr) {
        fpv_vehicle_connector_ = nullptr;
        CameraDirector = nullptr;
    }
    
    Super::EndPlay(EndPlayReason);
}

VehiclePawnWrapper* ASimModeWorldMultiRotor::getFpvVehiclePawnWrapper()
{
    return fpv_vehicle_pawn_wrapper_;
}

void ASimModeWorldMultiRotor::setupVehiclesAndCamera(std::vector<VehiclePtr>& vehicles)
{
    //get player controller
    APlayerController* player_controller = this->GetWorld()->GetFirstPlayerController();
    FTransform actor_transform = player_controller->GetActorTransform();
    //put camera little bit above vehicle
    FTransform camera_transform(actor_transform.GetLocation() + FVector(-300, 0, 200));
    
    //we will either find external camera if it already exist in evironment or create one
    APIPCamera* external_camera;
    
    //find all BP camera directors in the environment
    {
        TArray<AActor*> camera_dirs;
        UAirBlueprintLib::FindAllActor<ACameraDirector>(this, camera_dirs);
        if (camera_dirs.Num() == 0) {
            //create director
            FActorSpawnParameters camera_spawn_params;
            camera_spawn_params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;
            CameraDirector = this->GetWorld()->SpawnActor<ACameraDirector>(camera_director_class_, camera_transform, camera_spawn_params);
            CameraDirector->setFollowDistance(225);
            CameraDirector->setCameraRotationLagEnabled(false);
            CameraDirector->setFpvCameraIndex(0);
            CameraDirector->enableFlyWithMeMode();
            spawned_actors_.Add(CameraDirector);
            
            //create external camera required for the director
            external_camera = this->GetWorld()->SpawnActor<APIPCamera>(external_camera_class_, camera_transform, camera_spawn_params);
            spawned_actors_.Add(external_camera);
        }
        else {
            CameraDirector = static_cast<ACameraDirector*>(camera_dirs[0]);
            external_camera = CameraDirector->getExternalCamera();
        }
    }
    
    //find all vehicle pawns
    {
        TArray<AActor*> pawns;
        UAirBlueprintLib::FindAllActor<TMultiRotorPawn>(this, pawns);
        
        //if no vehicle pawns exists in environment
        if (pawns.Num() == 0) {
            //create vehicle pawn
            FActorSpawnParameters pawn_spawn_params;
            pawn_spawn_params.SpawnCollisionHandlingOverride =
            ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;
            TMultiRotorPawn* spawned_pawn = this->GetWorld()->SpawnActor<TMultiRotorPawn>(
                                                                                          vehicle_pawn_class_, actor_transform, pawn_spawn_params);
            
            spawned_actors_.Add(spawned_pawn);
            pawns.Add(spawned_pawn);
        }
        
        //set up vehicle pawns
        for (AActor* pawn : pawns)
        {
            //initialize each vehicle pawn we found
            TMultiRotorPawn* vehicle_pawn = static_cast<TMultiRotorPawn*>(pawn);
            vehicle_pawn->initializeForBeginPlay();
            
            //chose first pawn as FPV if none is designated as FPV
            VehiclePawnWrapper* wrapper = vehicle_pawn->getVehiclePawnWrapper();
            if (enable_collision_passthrough)
                wrapper->config.enable_passthrough_on_collisions = true;
            if (wrapper->config.is_fpv_vehicle || fpv_vehicle_pawn_wrapper_ == nullptr)
                fpv_vehicle_pawn_wrapper_ = wrapper;
            
            //now create the connector for each pawn
            VehiclePtr vehicle = createVehicle(wrapper);
            if (vehicle != nullptr) {
                vehicles.push_back(vehicle);
                
                if (fpv_vehicle_pawn_wrapper_ == wrapper)
                    fpv_vehicle_connector_ = vehicle;
            }
            //else we don't have vehicle for this pawn
        }
    }
    
    CameraDirector->initializeForBeginPlay(getInitialViewMode(), fpv_vehicle_pawn_wrapper_, external_camera);
}

void ASimModeWorldMultiRotor::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    getFpvVehiclePawnWrapper()->setLogLine(getLogString());
}

//sena was here
std::string ASimModeWorldMultiRotor::getLogString()
{
    const msr::airlib::Kinematics::State* kinematics = getFpvVehiclePawnWrapper()->getKinematics();
    msr::airlib::Vector3r_arr bones = getFpvVehiclePawnWrapper()->bones;
    uint64_t timestamp_millis = static_cast<uint64_t>(msr::airlib::ClockFactory::get()->nowNanos() / 1.0E6);
    
    //TODO: because this bug we are using alternative code with stringstream
    //https://answers.unrealengine.com/questions/664905/unreal-crashes-on-two-lines-of-extremely-simple-st.html
    
    std::string line;
    line.append(std::to_string(timestamp_millis)).append("\t")
    .append(std::to_string(bones.dronePos.x())).append("\t")
    .append(std::to_string(bones.dronePos.y())).append("\t")
    .append(std::to_string(bones.dronePos.z())).append("\t")
    .append(std::to_string(bones.droneOrient.x())).append("\t")
    .append(std::to_string(bones.droneOrient.y())).append("\t")
    .append(std::to_string(bones.droneOrient.z())).append("\t")
    .append(std::to_string(bones.hip.x())).append("\t")
    .append(std::to_string(bones.hip.y())).append("\t")
    .append(std::to_string(bones.hip.z())).append("\t")
    .append(std::to_string(bones.right_up_leg.x())).append("\t")
    .append(std::to_string(bones.right_up_leg.y())).append("\t")
    .append(std::to_string(bones.right_up_leg.z())).append("\t")
    .append(std::to_string(bones.right_leg.x())).append("\t")
    .append(std::to_string(bones.right_leg.y())).append("\t")
    .append(std::to_string(bones.right_leg.z())).append("\t")
    .append(std::to_string(bones.right_foot.x())).append("\t")
    .append(std::to_string(bones.right_foot.y())).append("\t")
    .append(std::to_string(bones.right_foot.z())).append("\t")
    .append(std::to_string(bones.left_up_leg.x())).append("\t")
    .append(std::to_string(bones.left_up_leg.y())).append("\t")
    .append(std::to_string(bones.left_up_leg.z())).append("\t")
    .append(std::to_string(bones.left_leg.x())).append("\t")
    .append(std::to_string(bones.left_leg.y())).append("\t")
    .append(std::to_string(bones.left_leg.z())).append("\t")
    .append(std::to_string(bones.left_foot.x())).append("\t")
    .append(std::to_string(bones.left_foot.y())).append("\t")
    .append(std::to_string(bones.left_foot.z())).append("\t")
    .append(std::to_string(bones.spine1.x())).append("\t")
    .append(std::to_string(bones.spine1.y())).append("\t")
    .append(std::to_string(bones.spine1.z())).append("\t")
    .append(std::to_string(bones.neck.x())).append("\t")
    .append(std::to_string(bones.neck.y())).append("\t")
    .append(std::to_string(bones.neck.z())).append("\t")
    .append(std::to_string(bones.head.x())).append("\t")
    .append(std::to_string(bones.head.y())).append("\t")
    .append(std::to_string(bones.head.z())).append("\t")
    .append(std::to_string(bones.left_arm.x())).append("\t")
    .append(std::to_string(bones.left_arm.y())).append("\t")
    .append(std::to_string(bones.left_arm.z())).append("\t")
    .append(std::to_string(bones.left_forearm.x())).append("\t")
    .append(std::to_string(bones.left_forearm.y())).append("\t")
    .append(std::to_string(bones.left_forearm.z())).append("\t")
    .append(std::to_string(bones.left_hand.x())).append("\t")
    .append(std::to_string(bones.left_hand.y())).append("\t")
    .append(std::to_string(bones.left_hand.z())).append("\t")
    .append(std::to_string(bones.right_arm.x())).append("\t")
    .append(std::to_string(bones.right_arm.y())).append("\t")
    .append(std::to_string(bones.right_arm.z())).append("\t")
    .append(std::to_string(bones.right_forearm.x())).append("\t")
    .append(std::to_string(bones.right_forearm.y())).append("\t")
    .append(std::to_string(bones.right_forearm.z())).append("\t")
    .append(std::to_string(bones.right_hand.x())).append("\t")
    .append(std::to_string(bones.right_hand.y())).append("\t")
    .append(std::to_string(bones.right_hand.z())).append("\t")
    .append(std::to_string(bones.right_hand_tip.x())).append("\t")
    .append(std::to_string(bones.right_hand_tip.y())).append("\t")
    .append(std::to_string(bones.right_hand_tip.z())).append("\t")
    .append(std::to_string(bones.left_hand_tip.x())).append("\t")
    .append(std::to_string(bones.left_hand_tip.y())).append("\t")
    .append(std::to_string(bones.left_hand_tip.z())).append("\t")
    .append(std::to_string(bones.right_foot_tip.x())).append("\t")
    .append(std::to_string(bones.right_foot_tip.y())).append("\t")
    .append(std::to_string(bones.right_foot_tip.z())).append("\t")
    .append(std::to_string(bones.left_foot_tip.x())).append("\t")
    .append(std::to_string(bones.left_foot_tip.y())).append("\t")
    .append(std::to_string(bones.left_foot_tip.z())).append("\t");
    
    return line;
    
    //std::stringstream ss;
    //ss << timestamp_millis << "\t";
    //ss << kinematics.pose.position.x() << "\t" << kinematics.pose.position.y() << "\t" << kinematics.pose.position.z() << "\t";
    //ss << kinematics.pose.orientation.w() << "\t" << kinematics.pose.orientation.x() << "\t" << kinematics.pose.orientation.y() << "\t" << kinematics.pose.orientation.z() << "\t";
    //ss << "\n";
    //return ss.str();
}

void ASimModeWorldMultiRotor::createVehicles(std::vector<VehiclePtr>& vehicles)
{
    //find vehicles and cameras available in environment
    //if none available then we will create one
    setupVehiclesAndCamera(vehicles);
}

ASimModeWorldBase::VehiclePtr ASimModeWorldMultiRotor::createVehicle(VehiclePawnWrapper* wrapper)
{
    auto vehicle_params = MultiRotorParamsFactory::createConfig(
                                                                wrapper->config.vehicle_config_name == "" ? default_vehicle_config
                                                                : std::string(TCHAR_TO_UTF8(*wrapper->config.vehicle_config_name)));
    
    vehicle_params_.push_back(std::move(vehicle_params));
    
    std::shared_ptr<MultiRotorConnector> vehicle = std::make_shared<MultiRotorConnector>(
                                                                                         wrapper, vehicle_params_.back().get(), enable_rpc, api_server_address,
                                                                                         vehicle_params_.back()->getParams().api_server_port, manual_pose_controller);
    
    if (vehicle->getPhysicsBody() != nullptr)
        wrapper->setKinematics(&(static_cast<PhysicsBody*>(vehicle->getPhysicsBody())->getKinematics()));
    
    return std::static_pointer_cast<VehicleConnectorBase>(vehicle);
}


