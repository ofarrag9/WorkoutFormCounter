import React, { useState } from "react";
import { View, Text } from "react-native";

export const WorkoutCard = ({ poses }) => {
  const [count, setCount] = useState(0);
  const [statusDown, setStatusDown] = useState(false);

  if (poses != null && poses.length > 0) {
    const knee_y = poses[0].keypoints[10].y;
    const hip_y = poses[0].keypoints[2].y;

    //Hip is lower than Knee

    if (knee_y < hip_y) {
      setStatusDown(true);
    } else if (knee_y > hip_y && statusDown) {
      // hip is higher than knee
      setCount(count + 1);
      setStatusDown(false);
    }
  }
  return (
    <View>
      <Text>{count}</Text>
    </View>
  );
};