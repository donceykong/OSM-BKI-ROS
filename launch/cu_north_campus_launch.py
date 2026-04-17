"""Launch MCD node and OSM visualizer for CU North Campus. No GT; no static TF; pose index = scan id. Config from cu_north_campus.yaml."""

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
    pkg_arg = DeclareLaunchArgument('pkg', default_value='osm_bki', description='Package name')
    method_arg = DeclareLaunchArgument('method', default_value='osm_bki', description='Method name')
    dataset_arg = DeclareLaunchArgument('dataset', default_value='cu_north_campus', description='Dataset name')
    data_root_arg = DeclareLaunchArgument(
        'data_root', default_value='',
        description='Root data directory; if empty, use data_root from config or else package data dir'
    )
    return LaunchDescription([
        pkg_arg, method_arg, dataset_arg, data_root_arg,
        OpaqueFunction(function=launch_setup)
    ])


def launch_setup(context):
    method = context.launch_configurations.get('method', 'osm_bki')
    dataset = context.launch_configurations.get('dataset', 'cu_north_campus')
    data_root_override = context.launch_configurations.get('data_root', '')

    pkg_share_dir = get_package_share_directory('osm_bki')
    ws_root = os.path.abspath(os.path.join(pkg_share_dir, '..', '..', '..', '..'))
    pkg_src_dir = os.path.join(ws_root, 'src', 'OSM-BKI-ROS')
    if not os.path.isdir(os.path.join(pkg_src_dir, 'config')):
        pkg_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    method_config_path = os.path.join(pkg_src_dir, 'config', 'methods', f'{method}.yaml')
    osm_bki_config_path = os.path.join(pkg_src_dir, 'config', 'methods', 'osm_bki.yaml')
    data_config_path = os.path.join(pkg_src_dir, 'config', 'methods', 'cu_north_campus.yaml')
    data_dir_path = _data_dir_from_config(data_config_path, pkg_src_dir, dataset, data_root_override)
    config_datasets_dir = os.path.join(pkg_src_dir, 'config', 'datasets')
    rviz_config_path = os.path.join(pkg_src_dir, 'rviz', 'cu_north_campus_node.rviz')

    # Load geometry parameters from osm_bki.yaml
    geom_params = {}
    if os.path.isfile(osm_bki_config_path):
        try:
            with open(osm_bki_config_path) as f:
                raw = yaml.safe_load(f)
            params = raw.get('/**', {}).get('ros__parameters', raw.get('ros__parameters', {}))
            geom_params = params.get('osm_geometry_parameters', {})
        except Exception as e:
            print(f'[WARN] Failed to load osm_bki.yaml geometry parameters: {e}')

    # No calibration (identity); publish_static_tf and use_pose_index_as_scan_id come from config
    mcd_params = [
        {'dir': data_dir_path},
        {'calibration_file': ''},
        {'config_datasets_dir': config_datasets_dir},
        method_config_path,
        data_config_path
    ]
    # Add osm_bki.yaml geometry parameters (overrides any in dataset config)
    if geom_params:
        mcd_params.append({'osm_geometry_parameters': geom_params})

    mcd_node = Node(
        package='osm_bki',
        executable='mcd_node',
        name='mcd_node',
        output='screen',
        parameters=mcd_params
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    # OSM visualizer uses same config and data_dir from launch; use osm_bki.yaml geometry
    osm_params = [data_config_path, {'data_dir': data_dir_path, 'config_datasets_dir': config_datasets_dir}]
    if geom_params:
        osm_params.append({'osm_geometry_parameters': geom_params})

    osm_node = Node(
        package='osm_bki',
        executable='osm_visualizer_node',
        name='osm_visualizer_node',
        output='screen',
        parameters=osm_params
    )

    return [rviz_node, mcd_node, osm_node]
