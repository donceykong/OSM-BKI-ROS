"""Launch KITTI360 node and OSM visualizer. Config from kitti360.yaml (mapping + OSM viz)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
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
    pkg_arg = DeclareLaunchArgument('pkg', default_value='osm_bki', description='Package name')
    method_arg = DeclareLaunchArgument('method', default_value='osm_bki', description='Method name')
    dataset_arg = DeclareLaunchArgument('dataset', default_value='kitti360', description='Dataset name')
    data_config_arg = DeclareLaunchArgument(
        'data_config', default_value='kitti360',
        description='Data config name (without .yaml); e.g. kitti360, kitti360_no_height, kitti360_with_height'
    )
    data_root_arg = DeclareLaunchArgument(
        'data_root', default_value='',
        description='Root data directory; if empty, use data_root from config or else package data dir'
    )
    return LaunchDescription([
        pkg_arg, method_arg, dataset_arg, data_config_arg, data_root_arg,
        OpaqueFunction(function=launch_setup)
    ])


def launch_setup(context):
    method = context.launch_configurations.get('method', 'osm_bki')
    dataset = context.launch_configurations.get('dataset', 'kitti360')
    data_root_override = context.launch_configurations.get('data_root', '')

    pkg_share_dir = get_package_share_directory('osm_bki')
    ws_root = os.path.abspath(os.path.join(pkg_share_dir, '..', '..', '..', '..'))
    pkg_src_dir = os.path.join(ws_root, 'src', 'OSM-BKI-ROS')
    if not os.path.isdir(os.path.join(pkg_src_dir, 'config')):
        pkg_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    data_config = context.launch_configurations.get('data_config', 'kitti360')

    method_config_path = os.path.join(pkg_src_dir, 'config', 'methods', f'{method}.yaml')
    data_config_path = os.path.join(pkg_src_dir, 'config', 'methods', f'{data_config}.yaml')
    data_dir_path = _data_dir_from_config(data_config_path, pkg_src_dir, dataset, data_root_override)
    config_datasets_dir = os.path.join(pkg_src_dir, 'config', 'datasets')
    rviz_config_path = os.path.join(pkg_src_dir, 'rviz', 'kitti360_node.rviz')

    # Read visualize flag from data config to decide whether to launch RViz/OSM viz
    visualize = True
    if os.path.isfile(data_config_path):
        try:
            with open(data_config_path) as f:
                raw = yaml.safe_load(f)
            params = raw.get('/**', {}).get('ros__parameters', raw.get('ros__parameters', {}))
            visualize = params.get('visualize', True)
        except Exception:
            pass

    kitti360_params = [
        {'dir': data_dir_path},
        {'calibration_file': ''},
        {'config_datasets_dir': config_datasets_dir},
        method_config_path,
        data_config_path
    ]

    kitti360_node = Node(
        package='osm_bki',
        executable='kitti360_node',
        name='kitti360_node',
        output='screen',
        parameters=kitti360_params
    )

    nodes = [kitti360_node]

    if visualize:
        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', rviz_config_path],
            output='screen'
        )

        # OSM visualizer uses same config and data_dir from launch; config from src
        osm_node = Node(
            package='osm_bki',
            executable='osm_visualizer_node',
            name='osm_visualizer_node',
            output='screen',
            parameters=[data_config_path, {'data_dir': data_dir_path, 'config_datasets_dir': config_datasets_dir}]
        )
        nodes.extend([rviz_node, osm_node])
    else:
        # Headless mode: shut down launch when the main node exits
        nodes.append(RegisterEventHandler(
            OnProcessExit(
                target_action=kitti360_node,
                on_exit=[Shutdown(reason='kitti360_node finished processing')]
            )
        ))

    return nodes
