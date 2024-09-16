# -*- coding: utf-8 -*-

import random
from os import path

import numpy as np
from qgis.PyQt.QtCore import QVariant
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingProvider,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterFileDestination,
                       QgsFeatureSink,
                       QgsVectorLayer,
                       QgsFeature,
                       QgsGeometry,
                       QgsField,
                       QgsProject, QgsProcessingParameterFolderDestination,
                       QgsPointXY, QgsWkbTypes, QgsProcessingParameterFile)
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon, MultiPolygon


# Algorithm 1: Create Mesh
class CreateMeshAlgorithm(QgsProcessingAlgorithm):
    INPUT_LAYER = 'INPUT_LAYER'
    POINT_COUNT = 'POINT_COUNT'
    OUTPUT_LAYER = 'OUTPUT_LAYER'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT_LAYER, 'Input layer', [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterNumber(
            self.POINT_COUNT, 'Number of points', QgsProcessingParameterNumber.Integer, defaultValue=100))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_LAYER, 'Output Mesh layer', QgsProcessing.TypeVectorPolygon))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT_LAYER, context)
        point_count = self.parameterAsInt(parameters, self.POINT_COUNT, context)
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT_LAYER, context,
                                               source.fields(), source.wkbType(), source.sourceCrs())

        all_points = []
        polygons = []

        for feature in source.getFeatures():
            if feedback.isCanceled():
                return {}
            geom = feature.geometry()
            if geom.isMultipart():
                # For multipart features, treat each part as a separate polygon
                for part in geom.asMultiPolygon():
                    poly = Polygon(part[0])  # Assuming exterior ring
                    polygons.append(poly)
                    all_points.extend(part[0])
            else:
                poly = Polygon(geom.asPolygon()[0])  # Assuming exterior ring
                polygons.append(poly)
                all_points.extend(geom.asPolygon()[0])

        # Generate internal points within each polygon
        for poly in polygons:
            minx, miny, maxx, maxy = poly.bounds
            while len(all_points) < point_count:
                pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if poly.contains(pnt):
                    all_points.append((pnt.x, pnt.y))

        # Perform Delaunay triangulation
        if len(all_points) >= 3:
            points = np.array(all_points)
            triangulation = Delaunay(points)
            for simplex in triangulation.simplices:
                triangle_points = [QgsPointXY(points[i][0], points[i][1]) for i in simplex]
                triangle = QgsGeometry.fromPolygonXY([triangle_points])

                # Check if the centroid of the triangle is within any of the polygons
                centroid = triangle.centroid().asPoint()
                if any(poly.contains(Point(centroid.x(), centroid.y())) for poly in polygons):
                    feature = QgsFeature()
                    feature.setGeometry(triangle)
                    sink.addFeature(feature, QgsFeatureSink.FastInsert)

        return {self.OUTPUT_LAYER: dest_id}

    def shortHelpString(self):
        return (
            "This algorithm generates a vector-based mesh suitable for Computational Fluid Dynamics (CFD) simulations.\n\n"
            "The number of points determines the mesh resolution. A higher number of points will result in a more "
            "refined mesh.\n\n"
            "Inputs:\n"
            "- Number of points: Specifies the number of points used to create the mesh grid. Increasing the number of points leads to a finer mesh.\n\n"
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
            self.OUTPUT_FILE, 'Output .su2 file', fileFilter='SU2 file (*.su2)'))

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
                        tri_points = ring[:-1]
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

                            # Classify as inlet, outlet, wall or fluid
                            if inlet_layer and any(inlet_geom.contains(centroid) for inlet_geom in inlet_geometries):
                                inlet_elements.append(tuple(element))
                            elif outlet_layer and any(
                                    outlet_geom.contains(centroid) for outlet_geom in outlet_geometries):
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
        # Path to SU2 executable
        self.addParameter(QgsProcessingParameterFile(
            self.SU2_CFD_PATH, 'SU2 CFD Executable', fileFilter='*.exe'))  # Expecting an executable file (.exe)

        # Path to SU2 input .su2 file
        self.addParameter(QgsProcessingParameterFile(
            self.SU2_FILE, 'SU2 Input File', fileFilter='*.su2'))  # Expecting an input .su2 file

        # Path to SU2 configuration .cfg file
        self.addParameter(QgsProcessingParameterFile(
            self.SU2_CFG_FILE, 'SU2 Configuration File', fileFilter='*.cfg'))  # Expecting a config file (.cfg)

        # Output folder
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_FOLDER, 'Output Folder'))

    def processAlgorithm(self, parameters, context, feedback):
        # Retrieve inputs
        su2_cfd_path = self.parameterAsFile(parameters, self.SU2_CFD_PATH, context)
        su2_file = self.parameterAsFile(parameters, self.SU2_FILE, context)
        su2_cfg_file = self.parameterAsFile(parameters, self.SU2_CFG_FILE, context)
        output_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)

        # TODO: Update the cfg_file from su2_file

        # Construct and run the SU2 command
        feedback.pushInfo('Running SU2 CFD Simulation...')
        process = QProcess()

        # Wrap paths with double quotes to handle spaces in file paths
        command = [f'"{su2_cfd_path}"', f'"{su2_cfg_file}"']
        process.setWorkingDirectory(output_folder)
        process.start(command[0], command[1:])

        if not process.waitForStarted():
            raise QgsProcessingException('Could not start SU2 process.')

        # Wait for the process to finish
        process.waitForFinished()
        feedback.pushInfo('SU2 CFD Simulation completed.')

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
                "- Output folder where simulation results will be saved\n\n"
                "Once configured, the SU2 CFD executable will run the provided configuration file "
                "and generate simulation results in the specified output folder.")

    def createInstance(self):
        return RunSU2CFD()
