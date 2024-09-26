# -*- coding: utf-8 -*-
import glob
import os
import os
import random
import random
import random
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from qgis.PyQt.QtCore import QCoreApplication, QUrl
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterFileDestination,
                       QgsFeatureSink,
                       QgsFeature,
                       QgsGeometry, QgsProcessingException,
                       QgsProcessingParameterFolderDestination,
                       QgsPointXY, QgsProcessingParameterFile)
from qgis.core import QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterFile, QgsProcessingParameterString, \
    QgsProcessingParameterFolderDestination, QgsProcessingParameterNumber
from qgis.core import QgsProcessingAlgorithm, QgsProcessingParameterFeatureSource, QgsProcessingParameterNumber, \
    QgsProcessingParameterFeatureSink, QgsFeature, QgsPointXY, QgsGeometry, QgsFeatureSink
from qgis.core import QgsWkbTypes, QgsProcessingAlgorithm, QgsProcessingParameterFeatureSource, \
    QgsProcessingParameterNumber, QgsFields, \
    QgsProcessingParameterFeatureSink, QgsFeature, QgsPointXY, QgsGeometry, QgsFeatureSink
from scipy.spatial import Delaunay
from scipy.spatial import Delaunay
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon
from shapely.geometry import Polygon, Point
from shapely.geometry import Polygon, Point
from PyQt5.QtGui import QDesktopServices


class CreateMeshAlgorithm(QgsProcessingAlgorithm):
    INPUT_LAYER = 'INPUT_LAYER'
    POINT_COUNT = 'POINT_COUNT'
    OUTPUT_LAYER = 'OUTPUT_LAYER'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT_LAYER, 'Input layer', [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterNumber(
            self.POINT_COUNT, 'Number of nodes', QgsProcessingParameterNumber.Integer, defaultValue=100))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_LAYER, 'Output Mesh layer', QgsProcessing.TypeVectorPolygon))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT_LAYER, context)
        point_count = self.parameterAsInt(parameters, self.POINT_COUNT, context)

        # Create an empty QgsFields object for the sink
        fields = QgsFields()  # No fields, just geometry

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT_LAYER, context,
                                               fields,  # Empty fields object
                                               QgsWkbTypes.Polygon,  # Specify geometry type as Polygon
                                               source.sourceCrs())

        all_points = []
        polygons = []

        # Pre-calculate geometries and bounds for faster point generation
        for feature in source.getFeatures():
            if feedback.isCanceled():
                return {}
            geom = feature.geometry()
            if geom.isMultipart():
                for part in geom.asMultiPolygon():
                    poly = Polygon(part[0])
                    polygons.append((poly, poly.bounds))
                    all_points.extend(part[0])
            else:
                poly = Polygon(geom.asPolygon()[0])
                polygons.append((poly, poly.bounds))
                all_points.extend(geom.asPolygon()[0])

        # Cache existing point count to avoid unnecessary point generation
        existing_point_count = len(all_points)

        # Generate internal points within polygons only if more points are needed
        for poly, bounds in polygons:
            minx, miny, maxx, maxy = bounds
            while len(all_points) < existing_point_count + point_count:
                pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if poly.contains(pnt):
                    all_points.append((pnt.x, pnt.y))

        # Perform Delaunay triangulation
        if len(all_points) >= 3:
            points = np.array(all_points)
            triangulation = Delaunay(points)
            triangles_to_add = []

            for simplex in triangulation.simplices:
                triangle_points = [QgsPointXY(points[i][0], points[i][1]) for i in simplex]
                triangle = QgsGeometry.fromPolygonXY([triangle_points])
                centroid = triangle.centroid().asPoint()

                # Check if centroid is within any of the polygons (cached check)
                if any(poly.contains(Point(centroid.x(), centroid.y())) for poly, _ in polygons):
                    feature = QgsFeature()
                    feature.setGeometry(triangle)
                    triangles_to_add.append(feature)

            # Add features in bulk to the sink
            sink.addFeatures(triangles_to_add, QgsFeatureSink.FastInsert)

        return {self.OUTPUT_LAYER: dest_id}

    def shortHelpString(self):
        return (
            "This algorithm generates a vector-based mesh suitable for Computational Fluid Dynamics (CFD) simulations.\n\n"
            "The number of nodes determines the mesh resolution. A higher number of nodes will result in a more "
            "refined mesh.\n\n"
            "Inputs:\n"
            "- Number of nodes: Specifies the number of nodes used to create the mesh grid. Increasing the number of "
            "nodes leads to a finer mesh.\n\n"
            "Output:\n"
            "- A mesh layer that can be used in CFD simulations.")

    def name(self):
        return 'vector_mesh_creation'

    def displayName(self):
        return 'Create Vector Mesh'

    def createInstance(self):
        return CreateMeshAlgorithm()


# Algorithm 2: Export to .su2 File
class ExportMeshToSu2Algorithm(QgsProcessingAlgorithm):
    MESH_LAYER = 'MESH_LAYER'
    INLET_LAYER = 'INLET_LAYER'
    OUTLET_LAYER = 'OUTLET_LAYER'
    OUTPUT_FILE = 'OUTPUT_FILE'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.MESH_LAYER, 'Mesh Layer', [QgsProcessing.TypeVectorPolygon]))
        # Inlet Layer (Optional)
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INLET_LAYER, 'Inlet Layer', [QgsProcessing.TypeVectorPolygon], optional=True))

        # Outlet Layer (Optional)
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.OUTLET_LAYER, 'Outlet Layer', [QgsProcessing.TypeVectorPolygon], optional=True))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_FILE, 'Output SU2 file', fileFilter='SU2 file (*.su2)'))

    def processAlgorithm(self, parameters, context, feedback):
        mesh_layer = self.parameterAsVectorLayer(parameters, self.MESH_LAYER, context)
        inlet_layer = self.parameterAsVectorLayer(parameters, self.INLET_LAYER, context)
        outlet_layer = self.parameterAsVectorLayer(parameters, self.OUTLET_LAYER, context)
        output_file_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FILE, context)

        vertices, elements, inlet_elements, outlet_elements, wall_elements, fluid_elements = self.processLayers(
            inlet_layer, outlet_layer, mesh_layer)

        with open(output_file_path, 'w') as file:
            self.writeSU2File(file, vertices, elements, inlet_elements, outlet_elements, wall_elements, fluid_elements)

        return {self.OUTPUT_FILE: output_file_path}

    def processLayers(self, inlet_layer=None, outlet_layer=None, mesh_layer=None):
        # Initialize vertices as set and use tuples for faster membership testing
        vertices = set()
        vertex_map = {}  # To store index mapping later
        elements = []

        inlet_elements, outlet_elements, wall_elements, fluid_elements = [], [], [], []

        if inlet_layer:
            inlet_geometries = [f.geometry() for f in inlet_layer.getFeatures()]
            print(f"Inlet geometries: {len(inlet_geometries)} geometries found")
        if outlet_layer:
            outlet_geometries = [f.geometry() for f in outlet_layer.getFeatures()]

        # Batch processing the mesh layer
        for feature in mesh_layer.getFeatures():
            geometry = feature.geometry()
            polygons = geometry.asMultiPolygon() if geometry.isMultipart() else [geometry.asPolygon()]

            for polygon in polygons:
                for ring in polygon:
                    if len(ring) >= 4:
                        tri_points = ring[:-1]  # Ignore the last point (closed polygon)
                        element = []

                        for point in tri_points:
                            vertex = (point.x(), point.y())  # Use tuple for faster comparison
                            if vertex not in vertices:
                                vertices.add(vertex)  # Add unique vertex
                                vertex_map[vertex] = len(vertex_map) + 1  # Index mapping
                            element.append(vertex_map[vertex])

                        if len(element) == 3:
                            fluid_elements.append(tuple(element))

                            # Classify centroid here (inlet/outlet/etc.)
                            centroid = QgsGeometry.fromPolylineXY([QgsPointXY(tri_points[0]),
                                                                   QgsPointXY(tri_points[1]),
                                                                   QgsPointXY(tri_points[2])]).centroid().asPoint()

                            # Classify as inlet, outlet, wall or fluid based on centroid or polygon containment
                            triangle_geom = QgsGeometry.fromPolygonXY([tri_points])  # Create geometry of triangle

                            if inlet_layer and any(inlet_geom.contains(centroid) or inlet_geom.intersects(triangle_geom)
                                                   for inlet_geom in inlet_geometries):
                                inlet_elements.append(tuple(element))
                            elif outlet_layer and any(
                                    outlet_geom.contains(centroid) or outlet_geom.intersects(triangle_geom)
                                    for outlet_geom in outlet_geometries):
                                outlet_elements.append(tuple(element))
                            else:
                                wall_elements.append(tuple(element))

                        elements.append(tuple(element))

        return vertex_map, elements, inlet_elements, outlet_elements, wall_elements, fluid_elements

    def writeSU2File(self, file, vertices, elements, inlet_elements, outlet_elements, wall_elements, fluid_elements):
        # Writing the dimensions and the number of elements in the fluid domain
        file.write("NDIME= 2\n")
        file.write(f"NELEM= {len(fluid_elements)}\n")

        # Write the fluid elements (triangles)
        for index, element in enumerate(fluid_elements):
            if len(element) == 3:
                file.write(f"5 {element[0] - 1} {element[1] - 1} {element[2] - 1} {index}\n")
            else:
                raise ValueError(f"Fluid element does not have 3 points: {element}")

        # Write the vertices
        file.write(f"NPOIN= {len(vertices)}\n")
        for (x, y), index in vertices.items():
            file.write(f"{x} {y} {index}\n")

        # Boundary markers count
        file.write("NMARK= 5\n")

        # Writing inlet, outlet, wall, and other boundaries using the provided method
        self.writeBoundaryMarker(file, 'inlet', inlet_elements, fluid_elements)
        self.writeBoundaryMarker(file, 'outlet', outlet_elements, fluid_elements)
        self.writeBoundaryMarker(file, 'wall', wall_elements, fluid_elements)
        self.writeBoundaryMarker(file, 'fluid', fluid_elements, fluid_elements)  # Fluid domain elements
        self.writeBoundaryMarker(file, 'Domain', fluid_elements, fluid_elements)  # Domain boundary

    def writeBoundaryMarker(self, file, tag, surface_elements, volume_elements):
        file.write(f"MARKER_TAG= {tag}\n")

        # Create a set of volume element points for fast lookup
        volume_points = set()
        for volume_element in volume_elements:
            volume_points.update(volume_element)

        # Buffer for batch file writing
        marker_lines = []

        # Filter elements that are connected to the volume
        valid_elements = []
        for element in surface_elements:
            # Check if all points in the surface element exist in the volume_points
            if all(point in volume_points for point in element):
                valid_elements.append(element)

        marker_lines.append(f"MARKER_ELEMS= {len(valid_elements)}\n")

        # Prepare data to write in batch
        for element in valid_elements:
            if len(element) == 2:  # If it's a line element (e.g., for boundary)
                marker_lines.append(f"3 {element[0] - 1} {element[1] - 1}\n")
            elif len(element) == 3:  # For triangular fluid elements
                marker_lines.append(f"5 {element[0] - 1} {element[1] - 1} {element[2] - 1}\n")
            else:
                raise ValueError(f"Unexpected element length: {len(element)}")

        # Write all the marker lines in one go
        file.write(''.join(marker_lines))
        file.flush()

    def name(self):
        return 'export_su2'

    def shortHelpString(self):
        return (
            "This algorithm exports a mesh suitable for CFD simulations in .su2 format.\n\n"
            "Inputs:\n"
            "- **Inlet Layer**: Specifies the inlet zones (e.g., for air, water, or species transport) in the mesh.\n"
            "- **Outlet Layer**: Specifies the outlet zones in the mesh.\n"
            "- **Mesh Layer**: The mesh layer that was created using the 'Create Vector Mesh' algorithm. This will be exported in .su2 format.\n\n"
            "Output:\n"
            "- The algorithm exports the mesh to a .su2 file, which is compatible with SU2 CFD simulations."
        )

    def displayName(self):
        return 'Export SU2 File'

    def createInstance(self):
        return ExportMeshToSu2Algorithm()



class RunSU2CFD(QgsProcessingAlgorithm):
    SU2_CFD_PATH = 'SU2_CFD_PATH'  # SU2 executable path
    SU2_FILE = 'SU2_FILE'  # SU2 input .su2 file
    SU2_CFG_FILE = 'SU2_CFG_FILE'  # SU2 configuration .cfg file
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'  # Output folder

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.SU2_CFD_PATH, 'SU2 CFD Executable', extension='exe'))

        # Path to SU2 input .su2 file
        self.addParameter(QgsProcessingParameterFile(self.SU2_FILE, 'SU2 Input File', extension='su2'))

        # Path to SU2 configuration .cfg file
        self.addParameter(QgsProcessingParameterFile(self.SU2_CFG_FILE, 'SU2 Configuration File', extension='cfg'))

        # Output folder
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def processAlgorithm(self, parameters, context, feedback):
        # Retrieve inputs
        su2_cfd_path = self.parameterAsFile(parameters, self.SU2_CFD_PATH, context)
        su2_file = self.parameterAsFile(parameters, self.SU2_FILE, context)
        su2_cfg_file = self.parameterAsFile(parameters, self.SU2_CFG_FILE, context)
        output_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)

        # Debugging: Log the file paths
        feedback.pushInfo(f"SU2 CFD Executable Path: {su2_cfd_path}")
        feedback.pushInfo(f"SU2 Input File Path: {su2_file}")
        feedback.pushInfo(f"SU2 Config File Path: {su2_cfg_file}")
        feedback.pushInfo(f"Output Folder Path: {output_folder}")

        # Ensure paths are valid
        if not su2_cfd_path or not su2_file or not su2_cfg_file:
            raise QgsProcessingException('Invalid file paths provided.')

        # Copy the .su2 file to the output folder (where the subprocess will run)
        try:
            feedback.pushInfo(f'Copying .su2 file to output folder: {output_folder}')
            shutil.copy(su2_file, output_folder)
            feedback.pushInfo(f'.su2 file successfully copied to {output_folder}')
        except:
            pass

        # Set the current working directory to the output folder
        os.chdir(output_folder)

        # Construct and run the SU2 command
        feedback.pushInfo('Running SU2 CFD Simulation...')
        command = [su2_cfd_path, su2_cfg_file]

        try:
            # Run the process and capture output in real-time
            process = subprocess.Popen(command, cwd=output_folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, shell=True)

            # Read the output line by line and print to the feedback (QGIS Processing log)
            for line in process.stdout:
                feedback.pushInfo(line.strip())  # Print each line in real-time

            process.wait()  # Wait for the process to complete

            # Check if the process completed successfully
            if process.returncode != 0:
                raise QgsProcessingException(f"SU2 CFD process failed with return code {process.returncode}.")

            feedback.pushInfo('SU2 CFD Simulation completed successfully.')

        except Exception as e:
            raise QgsProcessingException(f"An error occurred while running SU2: {str(e)}")

        # Return the output folder as the result
        return {self.OUTPUT_FOLDER: output_folder}

    def name(self):
        return 'runsu2cfd'

    def displayName(self):
        return 'Run SU2 CFD'

    def shortHelpString(self):
        return ("This algorithm allows users to run SU2 CFD simulations directly from QGIS.\n\n"
                "Inputs:\n"
                "- SU2 executable path (.exe)\n"
                "- SU2 input file (.su2)\n"
                "- SU2 configuration file (.cfg)\n"
                "- Output folder where simulation results will be saved in CSV files.\n\n"
                "Once configured, the SU2 CFD executable will run the provided configuration file "
                "and generate simulation results in the specified output folder.")

    def createInstance(self):
        return RunSU2CFD()


class PollutantDistributionGifAlgorithm(QgsProcessingAlgorithm):
    # Define input parameters and output
    CSV_FOLDER = 'CSV_FOLDER'
    PARAMETER = 'PARAMETER'
    OUTPUT_GIF_FILE = 'OUTPUT_GIF_FILE'
    DURATION = 'DURATION'

    def initAlgorithm(self, config=None):
        # Add input for the folder containing CSV files
        self.addParameter(QgsProcessingParameterFile(
            self.CSV_FOLDER,
            self.tr('Directory containing CSV files'),
            behavior=QgsProcessingParameterFile.Folder
        ))

        # Add input for the parameter to be plotted, with examples
        self.addParameter(QgsProcessingParameterString(
            self.PARAMETER,
            self.tr('Parameter to plot (e.g., Species_0, Pressure, Velocity_x, Velocity_y, Residual_Species_0)')
        ))

        # Add input for the GIF output file
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_GIF_FILE, 'GIF Output File', fileFilter='GIF file (*.gif)'))

        # Add input for frame duration (optional)
        self.addParameter(QgsProcessingParameterNumber(
            self.DURATION,
            self.tr('Frame duration in milliseconds (default is 200ms)'),
            defaultValue=200
        ))

    def processAlgorithm(self, parameters, context, feedback):
        # Get input values
        csv_folder = self.parameterAsFile(parameters, self.CSV_FOLDER, context)
        parameter = self.parameterAsString(parameters, self.PARAMETER, context)
        output_gif_path = self.parameterAsFileOutput(parameters, self.OUTPUT_GIF_FILE, context)
        duration = self.parameterAsInt(parameters, self.DURATION, context)

        # Find CSV files in the specified folder
        csv_files_pattern = os.path.join(csv_folder, 'result_*.csv')
        csv_files = glob.glob(csv_files_pattern)

        if not csv_files:
            raise QgsProcessingException('No CSV files found in the specified folder')

        image_files = []

        def plot_pressure_from_csv(csv_file_path, parameter, output_image_path):
            data = pd.read_csv(csv_file_path)
            x = data['x']
            y = data['y']
            variable = data[parameter]

            plt.figure(figsize=(16, 16))
            scatter = plt.scatter(x, y, c=variable, cmap='jet_r', s=10, vmax=np.max(variable), vmin=np.min(variable))
            plt.title(f'{parameter} Distribution with Boundary')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

            # Add color bar as a legend
            cbar = plt.colorbar(scatter)
            cbar.set_label(parameter)  # Label the color bar with the parameter name

            plt.savefig(output_image_path)
            plt.close()

        # Iterate through the CSV files, create plots, and save them as images
        for i, csv_file in enumerate(csv_files):
            if i>4:
                feedback.pushInfo(f"Processing file {i + 1}/{len(csv_files)}: {os.path.basename(csv_file)}")
                image_file_path = os.path.join(os.path.dirname(output_gif_path), f'frame_{i}.png')
                image_files.append(image_file_path)

                if not os.path.exists(image_file_path):  # Avoid re-generating images
                    plot_pressure_from_csv(csv_file, parameter, image_file_path)

            feedback.setProgress(int((i + 1) / len(csv_files) * 100))

        # Create a GIF from the image files
        feedback.pushInfo("Creating GIF...")
        images = [Image.open(image) for image in image_files]
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )

        feedback.pushInfo(f"GIF created and saved as {output_gif_path}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_gif_path))
        feedback.pushInfo(f"The GIF has been opened automatically.")

        return {self.OUTPUT_GIF_FILE: output_gif_path}

    def name(self):
        return 'pollutant_distribution_gif'

    def displayName(self):
        return self.tr('Visualize Pollutant Distribution GIF')

    def tr(self, string):
        return QCoreApplication.translate(self.__class__.__name__, string)

    def createInstance(self):
        return PollutantDistributionGifAlgorithm()

