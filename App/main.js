import React, { useState, useEffect } from "react";
import { View, Text } from "react-native";
import { MainStack } from "./src/App.Nagivator";

import { NavigationContainer } from "@react-navigation/native";

export default function Main() {
  return (
    <NavigationContainer>
      <MainStack />
    </NavigationContainer>
  );
}