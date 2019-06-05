---
layout:     post
title:      ROS Cheat Sheat
subtitle:   Romantic Operating System Guid
date:       2018-11-07
author:     Yaodong Cui
header-img: img/post-bg-basketball.jpg
header-mask: 0.5
catalog: true
tags:
    - ROS
    - Cheat Sheet
---

# ROS Cheat Sheet

## Filesystem Command-line Tools

rospack/rosstack A tool inspecting packages/stacks.
roscd Changes directories to a package or
stack.
rosls Lists package or stack information.
roscreate-pkg Creates a new ROS package.
roscreate-stack Creates a new ROS stack.
rosdep Installs ROS package system dependencies.
rosmake Builds a ROS package.
roswtf Displays a errors and warnings about a
running ROS system or launch file.
rxdeps Displays package structure and depen-
dencies.

Usage:
$ rospack find [package]
$ roscd [package[/subdir]]
$ rosls [package[/subdir]]
$ roscreate-pkg [packagename]
$ rosmake [package]
$ rosdep install [package]
$ roswtf or roswtf [file]
$ rxdeps [options]

## Common Command-line Tools

## roscore

A collection of nodes and programs that are pre-requisites of a
ROS-based system. You must have a roscore running in order
for ROS nodes to communicate.

roscore is currently defined as:
master
parameter server
rosout

Usage:
$ roscore

## rosmsg/rossrv

rosmsg/rossrv displays Message/Service (msg/srv) data
structure definitions.
Commands:
rosmsg show Display the fields in the msg.
rosmsg users Search for code using the msg.
rosmsg md5 Display the msg md5 sum.
rosmsg package List all the messages in a package.
rosnode packages List all the packages with messages.

Examples:
Display the Pose msg:
$ rosmsg show Pose
List the messages in navmsgs:
$ rosmsg package navmsgs
List the files using sensormsgs/CameraInfo:
$ rosmsg users sensormsgs/CameraInfo

## rosrun

```
rosrun allows you to run an executable in an arbitrary package
without having to cd (or roscd) there first.
```
```
Usage:
$ rosrun package executable
```
```
Example:
Run turtlesim:
$ rosrun turtlesim turtlesimnode
```
## rosnode

```
Displays debugging information about ROS nodes, including
publications, subscriptions and connections.
```
```
Commands:
rosnode ping Test connectivity to node.
rosnode list List active nodes.
rosnode info Print information about a node.
rosnode machine List nodes running on a particular ma-
chine.
rosnode kill Kills a running node.
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
## roslaunch

```
Starts ROS nodes locally and remotely via SSH, as well as
setting parameters on the parameter server.
```
```
Examples:
Launch on a different port:
$ roslaunch -p 1234 package filename.launch
Launch a file in a package:
$ roslaunch package filename.launch
Launch on the local nodes:
$ roslaunch --local package filename.launch
```
## rostopic

```
A tool for displaying debug information about ROS topics,
including publishers, subscribers, publishing rate, and
messages.
```
```
Commands:
rostopic bw Display bandwidth used by topic.
rostopic echo Print messages to screen.
rostopic hz Display publishing rate of topic.
rostopic list Print information about active topics.
rostopic pub Publish data to topic.
rostopic type Print topic type.
rostopic find Find topics by type.
```
```
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
```
```
Commands:
rosparam set Set a parameter.
rosparam get Get a parameter.
rosparam load Load parameters from a file.
rosparam dump Dump parameters to a file.
rosparam delete Delete a parameter.
rosparam list List parameter names.
```
```
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
```
```
Commands:
rosservice list Print information about active services.
rosservice node Print the name of the node providing a
service.
rosservice call Call the service with the given args.
rosservice args List the arguments of a service.
rosservice type Print the service type.
rosservice uri Print the service ROSRPC uri.
rosservice find Find services by service type.
```
```
Examples:
Call a service from the command-line:
$ rosservice call /addtwoints 1 2
Pipe the output of rosservice to rossrv to view the srv type:
$ rosservice type addtwoints | rossrv show
Display all services of a particular type:
$ rosservice find rospytutorials/AddTwoInts
```

## Logging Command-line Tools

### rosbag

This is a set of tools for recording from and playing back to
ROS topics. It is intended to be high performance and avoids
deserialization and reserializationof the messages.

rosbag recordwill generate a “.bag” file (so named for
historical reasons) with the contents of all topics that you pass
to it.

Examples:
Record all topics:
$ rosbag record -a
Record select topics:
$ rosbag record topic1 topic

rosbag playwill take the contents of one or more bag file,
and play them back in a time-synchronized fashion.

Examples:
Replay all messages without waiting:
$ rosbag play -a demolog.bag
Replay several bag files at once:
$ rosbag play demo1.bag demo2.bag

## Graphical Tools

### rxgraph

Displays a graph of the ROS nodes that are currently running,
as well as the ROS topics that connect them.

Usage:
$ rxgraph

### rxplot

A tool for plotting data from one or more ROS topic fields
using matplotlib.

Examples:
To graph the data in different plots:
$ rxplot /topic1/field1 /topic2/field
To graph the data all on the same plot:
$ rxplot /topic1/field1,/topic2/field
To graph multiple fields of a message:
$ rxplot /topic1/field1:field2:field

### rxbag

```
A tool for visualizing, inspecting, and replaying histories (bag
files) of ROS messages.
```
```
Usage:
$ rxbag bagfile.bag
```
### rxconsole

```
A tool for displaying and filtering messages published on
rosout.
```
```
Usage:
$ rxconsole
```
## tf Command-line Tools

### tfecho

```
A tool that prints the information about a particular
transformation between a sourceframe and a targetframe.
```
```
Usage:
$ rosrun tf tfecho <sourceframe> <targetframe>
```
```
Examples:
To echo the transform between /map and /odom:
$ rosrun tf tfecho /map /odom
```
### viewframes

```
A tool for visualizing the full tree of coordinate transforms.
```
```
Usage:
$ rosrun tf viewframes
$ evince frames.pdf
```
```
