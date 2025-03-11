#!/usr/bin/env python3

# import rospy
# from nav_msgs.msg import Path
# from geometry_msgs.msg import PointStamped, PoseStamped
# from visualization_msgs.msg import Marker

# import sys


# def clear_all():
#     m = Marker()
#     m.header.stamp = rospy.Time.now();
#     m.header.frame_id = FRAME_ID
#     # Delete all
#     m.action = 3
#     vis_pub.publish(m)

# def redraw_path():
#     m = Marker()
#     m.header.stamp = rospy.Time.now();
#     m.header.frame_id = FRAME_ID
#     m.action = Marker.ADD
#     m.color.r = 0
#     m.color.g = 0
#     m.color.b = 1
#     m.color.a = 1
#     # Important!
#     m.scale.x = 0.1
#     m.type = Marker.LINE_LIST
#     prev_point = None
#     for p in path.poses:
#         if not prev_point:
#             prev_point = p
#             continue
#         m.points.append(prev_point.pose.position)
#         m.points.append(p.pose.position)
#         prev_point = p
#     vis_pub.publish(m)

# # PointStamped
# def point_callback(msg):
#     ps = PoseStamped()
#     ps.pose.position = msg.point
#     ps.header = msg.header
#     path.poses.append(ps)

#     rospy.loginfo('Got point:')
#     rospy.loginfo('  x: {0}'.format(msg.point.x))
#     rospy.loginfo('  y: {0}'.format(msg.point.y))

#     clear_all()
#     redraw_path()

# # PoseStamped
# def goal_callback(msg):
#     rospy.loginfo('Got goal')

#     clear_all()

#     path.poses.append(msg)
#     path.header.stamp = rospy.Time.now()
#     path.header.frame_id = FRAME_ID
#     path_pub.publish(path)

#     del path.poses[:]



# if __name__ != '__main__':
#     sys.exit(0)

# FRAME_ID = '/map'

# path = Path()
# path_pub = rospy.Publisher('path', Path, queue_size=100)
# vis_pub = rospy.Publisher('visualization_marker', Marker, queue_size=1)


# rospy.init_node('pathmaker')

# rospy.Subscriber('clicked_point', PointStamped, point_callback)
# rospy.Subscriber('move_base_simple/goal', PoseStamped, goal_callback)

# rospy.loginfo('Waiting for points. Use "Publish Point" in rviz')

# rospy.spin()



#!/usr/bin/env python

import rospy
import math
from nav_msgs.msg import Path,Odometry
from geometry_msgs.msg import PoseStamped,PointStamped

class showPath:
     def __init__(self):
        self.pose_stamped=None
        self.path_pub=None
        self.path=None
    
     def odom_callback(self,msg):
        self.pose_stamped = PoseStamped()
        self.pose_stamped.pose = msg.pose.pose
        self.pose_stamped.header.stamp = rospy.Time.now()
        self.pose_stamped.header.frame_id = "odom"  # 这里假设机器人的位姿是在 "map" 坐标系下

        self.path.poses.append(self.pose_stamped)
        self.path.header = self.pose_stamped.header
        self.path_pub.publish(self.path)
        # rospy.Subscriber('/clicked_point',PointStamped,clicked_callback)

    # def clicked_callback(msg):
        
    #     target_distance=math.sqrt((pose_stamped.pose.position.x - msg.point.x)**2+(pose_stamped.pose.position.y -msg.point.y)**2)
    #     if (target_distance<=0.13):
    #         path.poses.clear()
        
     def main(self):
            rospy.init_node('show_path', anonymous=True)
            self.path_pub = rospy.Publisher('path', Path, queue_size=10)
            rospy.Subscriber('/odom', Odometry, self.odom_callback)  # 假设里程计话题是 "/odom"
            self.path = Path()
            rate = rospy.Rate(10)  # 发布频率为 10Hz

            while not rospy.is_shutdown():
                rospy.spin()
                rate.sleep()
     def empty_path(self):
         self.path_pub = rospy.Publisher('path', Path, queue_size=10)
         self.path=Path()
         self.path_pub.publish(self.path)
     def path_callback(self,msg):
         msg.poses.clear() 
         self.path=Path() 
         self.path_pub(self.path)
         
     def clear_path(self):
         
        #  self.path = rospy.Subscriber('/path', Path,self.path_callback)
         self.path_pub = rospy.Publisher('path', Path, queue_size=10)
         self.path=Path() 
         self.path_pub.publish(self.path)
       
if __name__ == '__main__':
    print_path=showPath()
    print_path.main()
    
