"""Launch KITTI360 node and OSM visualizer. Config from kitti360.yaml (mapping + OSM viz)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
import os
import yaml
from ament_index_python.packages import get_package_share_directory


def _data_dir_from_config(data_config_path, pkg_src_dir, dataset, data_root_override=None):
    """Resolve data dir: data_root_override or data_root in config; if empty use pkg_src_dir/data/<dataset>."""
    local_data_dir = os.path.join(pkg_src_dir, 'data', dataset)
    data_root = (data_root_override or '').strip()
    if not data_root and os.path.isfile(data_config_path):
        try:
            with open(data_config_path) as f:
                raw = yaml.safe_load(f)
            params = raw.get('/**', {}).get('ros__parameters', raw.get('ros__parameters', {}))
            data_root = (params.get('data_root') or '').strip()
        except Exception:
            pass
    if not data_root:
        return local_data_dir
    return os.path.join(data_root, dataset)


def generate_launch_description():
    pkg_arg = DeclareLaunchArgument('pkg', default_value='semantic_bki', description='Package name')
    method_arg = DeclareLaunchArgument('method', default_value='semantic_bki', description='Method name')
    dataset_arg = DeclareLaunchArgument('dataset', default_value='kitti360', description='Dataset name')
    data_root_arg = DeclareLaunchArgument(
        'data_root', default_value='',
        description='Root data directory; if empty, use data_root from config or else package data dir'
    )
    return LaunchDescription([
        pkg_arg, method_arg, dataset_arg, data_root_arg,
        OpaqueFunction(function=launch_setup)
    ])


def launch_setup(context):
    method = context.launch_configurations.get('method', 'semantic_bki')
    dataset = context.launch_configurations.get('dataset', 'kitti360')
    data_root_override = context.launch_configurations.get('data_root', '')

    pkg_share_dir = get_package_share_directory('semantic_bki')
    ws_root = os.path.abspath(os.path.join(pkg_share_dir, '..', '..', '..', '..'))
    pkg_src_dir = os.path.join(ws_root, 'src', 'OSM-BKI-ROS')
    if not os.path.isdir(os.path.join(pkg_src_dir, 'config')):
        pkg_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    method_config_path = os.path.join(pkg_src_dir, 'config', 'methods', f'{method}.yaml')
    data_config_path = os.path.join(pkg_src_dir, 'config', 'methods', 'kitti360.yaml')
    data_dir_path = _data_dir_from_config(data_config_path, pkg_src_dir, dataset, data_root_override)
    rviz_config_path = os.path.join(pkg_src_dir, 'rviz', 'kitti360_node.rviz')

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

    # OSM visualizer uses same config (kitti360.yaml) and data_dir from launch
    osm_node = Node(
        package='semantic_bki',
        executable='osm_visualizer_node',
        name='osm_visualizer_node',
        output='screen',
        parameters=[data_config_path, {'data_dir': data_dir_path}]
    )

    return [rviz_node, kitti360_node, osm_node]
