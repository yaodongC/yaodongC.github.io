---
layout:     post
title:      ROS  Catkin Cheat Sheat
subtitle:   Romantic Operating System Guid
date:       2018-12-20
author:     Yaodong Cui
header-img: img/post-bg-gears.jpg
header-mask: 0.5
catalog: true
tags:
    - ROS
    - Cheat Sheet
---

# ROS Indigo Cheatsheet

## Filesystem Management Tools

rospack A tool for inspecting packages.
rospack profile Fixes path and pluginlib problems.
roscd Change directory to a package.
rospd/rosd Pushd equivalent for ROS.
rosls Lists package or stack information.
rosed Open requested ROS file in a text editor.
roscp Copy a file from one place to another.
rosdep Installs package system dependencies.
roswtf Displays a errors and warnings about a
running ROS system or launch file.
catkincreatepkg Creates a new ROS stack.
wstool Manage many repos in workspace.
catkinmake Builds a ROS catkin workspace.
rqtdep Displays package structure and depen-
dencies.

Usage:
$ rospack find [package]
$ roscd [package[/subdir]]
$ rospd [package[/subdir] | +N | -N]
$ rosd
$ rosls [package[/subdir]]
$ rosed [package] [file]
$ roscp [package] [file] [destination]
$ rosdep install [package]
$ roswtf or roswtf [file]
$ catkincreatepkg [packagename] [depend1]..[dependN]
$ wstool [init | set | update]
$ catkinmake
$ rqtdep [options]

## Start-up and Process Launch Tools

## roscore

The basis nodes and programs for ROS-based systems. A
roscore must be running for ROS nodes to communicate.

Usage:
$ roscore

## rosrun

Runs a ROS package’s executable with minimal typing.

Usage:
$ rosrun packagename executablename

Example (runs turtlesim):
$ rosrun turtlesim turtlesimnode

## roslaunch

Starts a roscore (if needed), local nodes, remote nodes via
SSH, and sets parameter server parameters.

Examples:
Launch a file in a package:
$ roslaunch packagename filename.launch
Launch on a different port:
$ roslaunch -p 1234 packagename filename.launch
Launch on the local nodes:
$ roslaunch --local packagename filename.launch

## Logging Tools

## rosbag

```
A set of tools for recording and playing back of ROS topics.
Commands:
rosbag record Record a bag file with specified topics.
rosbag play Play content of one or more bag files.
rosbag compress Compress one or more bag files.
rosbag decompress Decompress one or more bag files.
rosbag filter Filter the contents of the bag.
```
```
Examples:
Record select topics:
$ rosbag record topic1 topic
Replay all messages without waiting:
$ rosbag play -a demolog.bag
Replay several bag files at once:
$ rosbag play demo1.bag demo2.bag
```
## Introspection and Command Tools

## rosmsg/rossrv

```
Displays Message/Service (msg/srv) data structure definitions.
Commands:
rosmsg show Display the fields in the msg/srv.
rosmsg list Display names of all msg/srv.
rosmsg md5 Display the msg/srv md5 sum.
rosmsg package List all the msg/srv in a package.
rosmsg packages List all packages containing the msg/srv.
```
```
Examples:
Display the Pose msg:
$ rosmsg show Pose
List the messages in the navmsgs package:
$ rosmsg package navmsgs
List the packages using sensormsgs/CameraInfo:
$ rosmsg packages sensormsgs/CameraInfo
```
## rosnode

```
Displays debugging information about ROS nodes, including
publications, subscriptions and connections.
Commands:
rosnode ping Test connectivity to node.
rosnode list List active nodes.
rosnode info Print information about a node.
rosnode machine List nodes running on a machine.
rosnode kill Kill a running node.
```
```
Examples:
Kill all nodes:
$ rosnode kill -a
List nodes on a machine:
$ rosnode machine aqy.local
Ping all nodes:
$ rosnode ping --all
```
## rostopic

```
A tool for displaying information about ROS topics, including
publishers, subscribers, publishing rate, and messages.
Commands:
rostopic bw Display bandwidth used by topic.
rostopic echo Print messages to screen.
rostopic find Find topics by type.
rostopic hz Display publishing rate of topic.
rostopic info Print information about an active topic.
rostopic list List all published topics.
rostopic pub Publish data to topic.
rostopic type Print topic type.
Examples:
Publish hello at 10 Hz:
$ rostopic pub -r 10 /topicname stdmsgs/String hello
Clear the screen after each message is published:
$ rostopic echo -c /topicname
Display messages that match a given Python expression:
$ rostopic echo --filter "m.data==’foo’" /topicname
Pipe the output of rostopic to rosmsg to view the msg type:
$ rostopic type /topicname | rosmsg show
```
## rosparam

```
A tool for getting and setting ROS parameters on the
parameter server using YAML-encoded files.
Commands:
rosparam set Set a parameter.
rosparam get Get a parameter.
rosparam load Load parameters from a file.
rosparam dump Dump parameters to a file.
rosparam delete Delete a parameter.
rosparam list List parameter names.
Examples:
List all the parameters in a namespace:
$ rosparam list /namespace
Setting a list with one as a string, integer, and float:
$ rosparam set /foo "[’1’, 1, 1.0]"
Dump only the parameters in a specific namespace to file:
$ rosparam dump dump.yaml /namespace
```
## rosservice

```
A tool for listing and querying ROS services.
Commands:
rosservice list Print information about active services.
rosservice node Print name of node providing a service.
rosservice call Call the service with the given args.
rosservice args List the arguments of a service.
rosservice type Print the service type.
rosservice uri Print the service ROSRPC uri.
rosservice find Find services by service type.
Examples:
Call a service from the command-line:
$ rosservice call /addtwoints 1 2
Pipe the output of rosservice to rossrv to view the srv type:
$ rosservice type addtwoints | rossrv show
Display all services of a particular type:
$ rosservice find rospytutorials/AddTwoInts
```

# ROS Indigo Cheatsheet

## Logging Tools

## rqtconsole

A tool to display and filtering messages published on rosout.

Usage:
$ rqtconsole

## rqtbag

A tool for visualizing, inspecting, and replaying bag files.

Usage, viewing:
$ rqtbag bagfile.bag
Usage, bagging:
$ rqtbag *press the big red record button.*

## rqtloggerlevel

Change the logger level of ROS nodes. This will increase or
decrease the information they log to the screen and rqtconsole.

Usage:
viewing $ rqtloggerlevel

## Introspection & Command Tools

## rqttopic

A tool for viewing published topics in real time.

Usage:
$ rqt
Plugin Menu->Topic->Topic Monitor

## rqtmsg, rqtsrv, and rqtaction

A tool for viewing available msgs, srvs, and actions.

Usage:
$ rqt
Plugin Menu->Topic->Message Type Browser
Plugin Menu->Service->Service Type Browser
Plugin Menu->Action->Action Type Browser

## rqtpublisher, and rqtservicecaller

```
Tools for publishing messages and calling services.
Usage:
$ rqt
Plugin Menu->Topic->Message Publisher
Plugin Menu->Service->Service Caller
```
## rqtgraph, and rqtdep

```
Tools for displaying graphs of running ROS nodes with
connecting topics and package dependancies respectively.
```
```
Usage:
$ rqtgraph
$ rqtdep
```
## rqttop

```
A tool for ROS specific process monitoring.
Usage:
$ rqt
Plugin Menu->Introspection->Process Monitor
```
## rqtreconfigure

```
A tool for dynamically reconfiguring ROS parameters.
Usage:
$ rqt
Plugin Menu->Configuration->Dynamic Reconfigure
```
## Development Environments

## rqtshell, and rqtpyconsole

```
Two tools for accessing an xterm shell and python console
respectively.
Usage:
$ rqt
Plugin Menu->Miscellaneous Tools->Shell
Plugin Menu->Miscellaneous Tools->Python Console
```
## Data Visualization Tools

## tfecho

```
A tool that prints the information about a particular
transformation between a sourceframe and a targetframe.
Usage:
$ rosrun tf tfecho <sourceframe> <targetframe>
```
```
Examples:
To echo the transform between /map and /odom:
$ rosrun tf tfecho /map /odom
```
## viewframes

```
A tool for visualizing the full tree of coordinate transforms.
Usage:
$ rosrun tf2tools viewframes.py
$ evince frames.pdf
```
## rqtplot

```
A tool for plotting data from ROS topic fields.
```
```
Examples:
To graph the data in different plots:
$ rqtplot /topic1/field1 /topic2/field
To graph the data all on the same plot:
$ rqtplot /topic1/field1,/topic2/field
To graph multiple fields of a message:
$ rqtplot /topic1/field1:field2:field
```
## rqtimageview

```
A tool to display image topics.
```
```
Usage:
$ rqtimageview
```

## ROS Indigo Catkin Workspaces

### Create a catkin workspace

Setup and use a new catkin workspace from scratch.

Example:
$ source /opt/ros/hydro/setup.bash
$ mkdir -p ~/catkinws/src
$ cd ~/catkinws/src
$ catkininitworkspace

### Checkout an existing ROS package

Get a local copy of the code for an existing package and keep
it up to date using wstool.

Examples:
$ cd ~/catkinws/src
$ wstool init
$ wstool set tutorials --git git://github.com/ros/rostutorials.git
$ wstool update

### Create a new catkin ROS package

Create a new ROS catkin package in an existing workspace
with catkin create package. After using this you will need to
edit the CMakeLists.txt to detail how you want your package
built and add information to your package.xml.

Usage:
$ catkincreatepkg <packagename> [depend1] [depend2]

Example:
$ cd ~/catkinws/src
$ catkincreatepkg tutorials stdmsgs rospy roscpp

### Build all packages in a workspace

Use catkinmake to build all the packages in the workspace
and then source the setup.bash to add the workspace to the
ROSPACKAGEPATH.

Examples:
$ cd ~/catkinws
$ ~/catkinmake
$ source devel/setup.bash
