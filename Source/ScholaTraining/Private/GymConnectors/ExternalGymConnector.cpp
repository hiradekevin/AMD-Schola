// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "GymConnectors/ExternalGymConnector.h"
#include "LogScholaTraining.h"

UExternalGymConnector::UExternalGymConnector()
{
}


FTrainingStateUpdate* UExternalGymConnector::ResolveEnvironmentStateUpdate()
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaTraining: ExternalGymConnector Resolve Environment State Update");
	if (!PendingUpdateFuture.IsValid())
	{
		PendingUpdateFuture = this->RequestStateUpdate();
	}
	if (PendingUpdateFuture.WaitFor(FTimespan::Zero()))
	{
		FTrainingStateUpdate* Result = PendingUpdateFuture.Get();
		PendingUpdateFuture = TFuture<FTrainingStateUpdate*>();
		return Result;
	}
	return nullptr;
}
