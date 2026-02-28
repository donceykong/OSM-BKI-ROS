from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    pkg_arg = DeclareLaunchArgument(
        'pkg',
        default_value='semantic_bki',
        description='Package name'
    )
    
    method_arg = DeclareLaunchArgument(
        'method',
        default_value='semantic_bki',
        description='Method name'
    )
    
    dataset_arg = DeclareLaunchArgument(
        'dataset',
        default_value='sim_unstructured_annotated',
        description='Dataset name'
    )
    
    return LaunchDescription([
        pkg_arg,
        method_arg,
        dataset_arg,
        OpaqueFunction(function=launch_setup)
    ])


def launch_setup(context):
    # Get launch argument values
    method = context.launch_configurations.get('method', 'semantic_bki')
    dataset = context.launch_configurations.get('dataset', 'sim_unstructured_annotated')
    
    # Get package share directory
    pkg_share_dir = get_package_share_directory('semantic_bki')
    
    # Construct paths
    method_config_path = os.path.join(pkg_share_dir, 'config', 'methods', f'{method}.yaml')
    data_config_path = os.path.join(pkg_share_dir, 'config', 'datasets', f'{dataset}.yaml')
    data_dir_path = os.path.join(pkg_share_dir, 'data', dataset)
    rviz_config_path = os.path.join(pkg_share_dir, 'rviz', 'toy_example_node.rviz')
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    
    # Toy example node
    toy_example_node = Node(
        package='semantic_bki',
        executable='toy_example_node',
        name='toy_example_node',
        output='screen',
        parameters=[
            {'dir': data_dir_path},
            {'prefix': dataset},
            method_config_path,
            data_config_path
        ]
    )
    
    return [rviz_node, toy_example_node]
