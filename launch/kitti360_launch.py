"""Launch KITTI360 node (and optional OSM visualizer). Uses velodyne_poses.txt format; OSM in sequence dir; no static TF."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_arg = DeclareLaunchArgument('pkg', default_value='semantic_bki', description='Package name')
    method_arg = DeclareLaunchArgument('method', default_value='semantic_bki', description='Method name')
    dataset_arg = DeclareLaunchArgument('dataset', default_value='kitti360', description='Dataset name')
    osm_dataset_arg = DeclareLaunchArgument(
        'osm_dataset', default_value='osm_visualizer_kitti360',
        description='OSM visualizer config (use osm_visualizer_kitti360 for KITTI360 poses)'
    )
    return LaunchDescription([
        pkg_arg, method_arg, dataset_arg, osm_dataset_arg,
        OpaqueFunction(function=launch_setup)
    ])


def launch_setup(context):
    method = context.launch_configurations.get('method', 'semantic_bki')
    dataset = context.launch_configurations.get('dataset', 'kitti360')
    osm_dataset = context.launch_configurations.get('osm_dataset', 'osm_visualizer_kitti360')

    pkg_share_dir = get_package_share_directory('semantic_bki')
    ws_root = os.path.abspath(os.path.join(pkg_share_dir, '..', '..', '..', '..'))
    pkg_src_dir = os.path.join(ws_root, 'src', 'OSM-BKI-ROS')
    if not os.path.isdir(os.path.join(pkg_src_dir, 'config')):
        pkg_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    method_config_path = os.path.join(pkg_src_dir, 'config', 'methods', f'{method}.yaml')
    data_config_path = os.path.join(pkg_src_dir, 'config', 'methods', 'kitti360.yaml')
    osm_config_path = os.path.join(pkg_src_dir, 'config', 'datasets', f'{osm_dataset}.yaml')
    data_dir_path = os.path.join(pkg_src_dir, 'data', dataset)
    rviz_config_path = os.path.join(pkg_src_dir, 'rviz', 'kitti360_node.rviz')

    # KITTI360: no calibration file (identity); config has sequence_name and paths
    kitti360_params = [
        {'dir': data_dir_path},
        {'calibration_file': ''},
        method_config_path,
        data_config_path
    ]

    kitti360_node = Node(
        package='semantic_bki',
        executable='kitti360_node',
        name='kitti360_node',
        output='screen',
        parameters=kitti360_params
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    nodes = [rviz_node, kitti360_node]
    if os.path.isfile(osm_config_path):
        osm_node = Node(
            package='semantic_bki',
            executable='osm_visualizer_node',
            name='osm_visualizer_node',
            output='screen',
            parameters=[osm_config_path, {'data_dir': data_dir_path}]
        )
        nodes.append(osm_node)

    return nodes
