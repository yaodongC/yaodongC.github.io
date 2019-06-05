---
layout: post
title: "The Kinematic Bicycle Model"
subtitle:   a 2-DOF handling model
date:       2019-06-05
author:     Yaodong Cui
header-img: img/post-bg-book1.jpg
header-mask: 0.5
catalog: true
tags:
    - Bicycle Model
    - Vechicle Dynamic
---

# Kinematic Bicycle Model

Kinematic Bicycle Model is a 2-DOF handling model(yaw-plane models). It captures the lateral/yaw or sideslip/yaw motions only, while assuming a constant forward speed.

<br>
<div  align="center">
    <img
    src="https://raw.githubusercontent.com/yaodongC/yaodongC.github.io/master/post_img/190605/Kinematic-bicycle-model-representation.png"
    width = "800" height = "400"></div>
    <div align="center">Fig.1  The Kinematic Bicycle Model.</div>
<br>

<br> Bicycle model could be linear or nonlinear. It's tire model could be linear or nonlinear. No braking/traction inputs is considered in this model.

<br> No roll DOF considered in this model, but lateral load transfers could be captured within the tire model.
