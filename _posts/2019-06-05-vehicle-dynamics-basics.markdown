---
layout: post
title: "Vehicle Dynamics Basics"
subtitle:   A Beginner's Guid to Vehicle Dynamics
date:       2019-06-05
author:     Yaodong Cui
header-img: img/post-bg-coffee2.jpg
header-mask: 0.5
catalog: true
tags:
    - Vechicle Dynamic
---

# Longitudinal direction
## Longitudinal direction is the forward moving direction of the vehicle

<br> There are two different ways of looking at the forward direction, one with respect to the vehicle body itself, and another with respect to a fixed reference point.  The former is often used when dealing with acceleration and velocity of the vehicle.  The latter is used when the location information of the vehicle with respect to a starting or an ending point is
desired.

# Lateral direction
## Lateral direction is the sideways moving direction of the vehicle

<br> Again, there are two ways of looking at the lateral direction, with respect to the vehicle and with respect to a fixed reference point.  Researchers often find this direction more interesting than the
longitudinal one since extreme values of lateral acceleration or lateral velocity can decrease vehicle stability and controllability.

# Tire slip angle
## Tire slip angle is the angle between the tire's heading and the actual direction of tire's travel.

The slip angle of a vehicle describes the ratio of forward and lateral velocities in the form of an angle and is normally represented by the symbol β (Beta). A example of tire slip angle is shown in Fig.1.
<br>
<div  align="center">
    <img
    src="https://raw.githubusercontent.com/yaodongC/yaodongC.github.io/master/post_img/190605/Wheel-Slip-Calculation-The-tire-slip-angle-a-is-defined-to-be-the-angle-between-the.png"
    width = "202" height = "256"></div>
  <div align="center">Fig.1  Tire slip angle.</div>
<br>

# Body-slip angle
## slip angle is the difference between the direction a vehicle is travelling and the direction that the body of the vehicle is pointing.

<br>
<div  align="center">
    <img
    src="https://raw.githubusercontent.com/yaodongC/yaodongC.github.io/master/post_img/190605/bodyslip.gif"
    width = "564" height = "173"></div>
  <div align="center">Fig.1  Tire slip angle.</div>
<br>

Body-slip angle is the angle between the X-axis and the velocity vector that represents the instantaneous vehicle velocity at that point along the path.
