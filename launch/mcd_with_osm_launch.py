"""Launch MCD node and OSM visualizer. Config from mcd.yaml (mapping + OSM viz). data_root in config overrides local data dir."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
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
    # Declare launch arguments
    pkg_arg = DeclareLaunchArgument(
        'pkg',
        default_value='osm_bki',
        description='Package name'
    )
    
    method_arg = DeclareLaunchArgument(
        'method',
        default_value='osm_bki',
        description='Method name'
    )
    
    dataset_arg = DeclareLaunchArgument(
        'dataset',
        default_value='mcd',
        description='Dataset name'
    )

    data_root_arg = DeclareLaunchArgument(
        'data_root',
        default_value='',
        description='Root data directory; if empty, use data_root from config or else package data dir'
    )

    color_mode_arg = DeclareLaunchArgument(
        'color_mode',
        default_value='semantic',
        description='Visualization color mode: semantic, osm_building, osm_road, osm_grassland, osm_tree, osm_parking, osm_fence, osm_blend (all priors blended)'
    )
    
    osm_file_arg = DeclareLaunchArgument(
        'osm_file',
        default_value='',
        description='Path to OSM file (relative to data dir or absolute) for voxel priors. Empty to disable.'
    )
    
    osm_origin_lat_arg = DeclareLaunchArgument(
        'osm_origin_lat',
        default_value='0.0',
        description='OSM origin latitude (degrees)'
    )
    
    osm_origin_lon_arg = DeclareLaunchArgument(
        'osm_origin_lon',
        default_value='0.0',
        description='OSM origin longitude (degrees)'
    )
    
    osm_decay_meters_arg = DeclareLaunchArgument(
        'osm_decay_meters',
        default_value='2.0',
        description='OSM prior decay distance in meters'
    )
    
    return LaunchDescription([
        pkg_arg,
        method_arg,
        dataset_arg,
        data_root_arg,
        color_mode_arg,
        osm_file_arg,
        osm_origin_lat_arg,
        osm_origin_lon_arg,
        osm_decay_meters_arg,
        OpaqueFunction(function=launch_setup)
    ])


def launch_setup(context):
    method = context.launch_configurations.get('method', 'osm_bki')
    dataset = context.launch_configurations.get('dataset', 'mcd')
    data_root_override = context.launch_configurations.get('data_root', '')
    color_mode = context.launch_configurations.get('color_mode', 'semantic')
    osm_file = context.launch_configurations.get('osm_file', '')
    osm_origin_lat = context.launch_configurations.get('osm_origin_lat', '0.0')
    osm_origin_lon = context.launch_configurations.get('osm_origin_lon', '0.0')
    osm_decay_meters = context.launch_configurations.get('osm_decay_meters', '2.0')

    pkg_share_dir = get_package_share_directory('osm_bki')
    ws_root = os.path.abspath(os.path.join(pkg_share_dir, '..', '..', '..', '..'))
    pkg_src_dir = os.path.join(ws_root, 'src', 'OSM-BKI-ROS')
    if not os.path.isdir(os.path.join(pkg_src_dir, 'config')):
        pkg_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    method_config_path = os.path.join(pkg_src_dir, 'config', 'methods', f'{method}.yaml')
    methods_datasets = ('mcd', 'cu_north_campus')
    data_config_path = (
        os.path.join(pkg_src_dir, 'config', 'methods', f'{dataset}.yaml')
        if dataset in methods_datasets
        else os.path.join(pkg_src_dir, 'config', 'datasets', f'{dataset}.yaml')
    )
    data_dir_path = _data_dir_from_config(data_config_path, pkg_src_dir, dataset, data_root_override)
    config_datasets_dir = os.path.join(pkg_src_dir, 'config', 'datasets')
    # cu_north_campus (and kitti360) use identity calibration; no base-frame TF
    calib_file_path = '' if dataset == 'cu_north_campus' else os.path.join(data_dir_path, 'hhs_calib.yaml')
    rviz_config_path = os.path.join(pkg_src_dir, 'rviz', 'mcd_node.rviz')
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    
    # MCD node (scan processing) - now with OSM prior support
    # color_mode, osm_decay_meters, osm_file, osm_origin_* come from dataset config (config/methods/mcd.yaml) by default
    mcd_params = [
        {'dir': data_dir_path},
        {'calibration_file': calib_file_path},
        {'config_datasets_dir': config_datasets_dir},
        method_config_path,
        data_config_path
    ]
    # Allow launch args to override OSM settings when explicitly provided (e.g. osm_file:=other.osm)
    if osm_file or osm_origin_lat != '0.0' or osm_origin_lon != '0.0':
        mcd_params.append({
            'osm_file': osm_file,
            'osm_origin_lat': float(osm_origin_lat),
            'osm_origin_lon': float(osm_origin_lon),
        })
    
    mcd_node = Node(
        package='osm_bki',
        executable='mcd_node',
        name='mcd_node',
        output='screen',
        parameters=mcd_params
    )
    
    # OSM visualizer uses same config (mcd.yaml) and data_dir from launch; config from src
    osm_node = Node(
        package='osm_bki',
        executable='osm_visualizer_node',
        name='osm_visualizer_node',
        output='screen',
        parameters=[data_config_path, {'data_dir': data_dir_path, 'config_datasets_dir': config_datasets_dir}]
    )
    
    return [rviz_node, mcd_node, osm_node]
