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
                       QgsProject,
                       QgsPointXY, QgsWkbTypes)
from qgis.PyQt.QtCore import QVariant
from scipy.spatial import Delaunay
import random
from os import path
import numpy as np
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
        return "This algorithm creates a mesh suitable for CFD simulations. " \
               "Number of points determine mesh size. More points refer to refined mesh."

    def name(self):
        return 'vector_mesh_creation'

    def displayName(self):
        return 'Create Vector Mesh'

    def createInstance(self):
        return CreateMeshAlgorithm()


# Algorithm 2: Export to .su2 File
class ExportMeshToSu2Algorithm(QgsProcessingAlgorithm):
    MESH_LAYER = 'MESH_LAYER'
    OBJECT_LAYER = 'OBJECT_LAYER'
    INLET_LAYER = 'INLET_LAYER'
    OUTLET_LAYER = 'OUTLET_LAYER'
    OUTPUT_FILE = 'OUTPUT_FILE'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.MESH_LAYER, 'Mesh Layer', [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.OBJECT_LAYER, 'Object Layer', [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INLET_LAYER, 'Inlet Layer', [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.OUTLET_LAYER, 'Outlet Layer', [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_FILE, 'Output .su2 file', fileFilter='SU2 file (*.su2)'))

    def processAlgorithm(self, parameters, context, feedback):
        mesh_layer = self.parameterAsVectorLayer(parameters, self.MESH_LAYER, context)
        object_layer = self.parameterAsVectorLayer(parameters, self.OBJECT_LAYER, context)
        inlet_layer = self.parameterAsVectorLayer(parameters, self.INLET_LAYER, context)
        outlet_layer = self.parameterAsVectorLayer(parameters, self.OUTLET_LAYER, context)
        output_file_path = self.parameterAsFileOutput(parameters, self.OUTPUT_FILE, context)

        vertices, elements, inlet_elements, outlet_elements, wall_elements, fluid_elements = self.processLayers(
            object_layer, inlet_layer, outlet_layer, mesh_layer)

        with open(output_file_path, 'w') as file:
            self.writeSU2File(file, vertices, elements, inlet_elements, outlet_elements, wall_elements, fluid_elements)

        return {self.OUTPUT_FILE: output_file_path}

    def processLayers(self, object_layer, inlet_layer, outlet_layer, mesh_layer):
        vertices = {}
        elements = []
        inlet_elements = []
        outlet_elements = []
        wall_elements = []
        fluid_elements = []

        # Convert inlet and outlet layer features to geometries for easy checking
        inlet_geometries = [feature.geometry() for feature in inlet_layer.getFeatures()]
        outlet_geometries = [feature.geometry() for feature in outlet_layer.getFeatures()]

        # Process object layer to get vertices and elements
        for feature in object_layer.getFeatures():
            geometry = feature.geometry()
            if geometry.isMultipart():
                polygons = geometry.asMultiPolygon()
            else:
                polygons = [geometry.asPolygon()]

            for polygon in polygons:
                for ring in polygon:
                    for i in range(len(ring) - 1):
                        point1, point2 = QgsPointXY(ring[i]), QgsPointXY(ring[i + 1])
                        if point1 not in vertices:
                            vertices[point1] = len(vertices) + 1
                        if point2 not in vertices:
                            vertices[point2] = len(vertices) + 1
                        element = (vertices[point1], vertices[point2])

                        centroid = QgsGeometry.fromPolylineXY([point1, point2]).centroid().asPoint()
                        if any(inlet_geom.contains(centroid) for inlet_geom in inlet_geometries):
                            inlet_elements.append(element)
                        elif any(outlet_geom.contains(centroid) for outlet_geom in outlet_geometries):
                            outlet_elements.append(element)
                        else:
                            wall_elements.append(element)
                        elements.append(element)

        # Process mesh layer to define fluid domain
        for feature in mesh_layer.getFeatures():
            geometry = feature.geometry()
            if geometry.isMultipart():
                polygons = geometry.asMultiPolygon()
            else:
                polygons = [geometry.asPolygon()]

            for polygon in polygons:
                for ring in polygon:
                    if len(ring) >= 4:  # A triangle in QGIS polygon format (including closing point)
                        tri_points = ring[:-1]  # Exclude the closing point
                        element = []
                        for point in tri_points:
                            qgs_point = QgsPointXY(point)
                            if qgs_point not in vertices:
                                vertices[qgs_point] = len(vertices) + 1
                            element.append(vertices[qgs_point])
                        if len(element) == 3:
                            fluid_elements.append(tuple(element))

        return vertices, elements, inlet_elements, outlet_elements, wall_elements, fluid_elements

    def writeSU2File(self, file, vertices, elements, inlet_elements, outlet_elements, wall_elements, fluid_elements):
        file.write("NDIME= 2\n")
        file.write(f"NELEM= {len(fluid_elements)}\n")
        for index, element in enumerate(fluid_elements):
            if len(element) == 3:
                file.write(f"5 {element[0] - 1} {element[1] - 1} {element[2] - 1} {index}\n")
            else:
                raise ValueError(f"Fluid element does not have 3 points: {element}")

        file.write(f"NPOIN= {len(vertices)}\n")
        for point, index in vertices.items():
            file.write(f"{point.x()} {point.y()} {index}\n")
        file.write("NMARK= 5\n")

        # Updated calls to writeBoundaryMarker with all required arguments
        self.writeBoundaryMarker(file, 'inlet', inlet_elements, fluid_elements)
        self.writeBoundaryMarker(file, 'outlet', outlet_elements, fluid_elements)
        self.writeBoundaryMarker(file, 'wall', wall_elements, fluid_elements)
        self.writeBoundaryMarker(file, 'fluid', fluid_elements, fluid_elements)
        self.writeBoundaryMarker(file, 'Domain', fluid_elements, fluid_elements)

    def writeBoundaryMarker(self, file, tag, surface_elements, volume_elements):
        file.write(f"MARKER_TAG= {tag}\n")
        valid_elements = [element for element in surface_elements if self.isConnectedToVolume(element, volume_elements)]
        file.write(f"MARKER_ELEMS= {len(valid_elements)}\n")
        for element in valid_elements:
            if len(element) == 2:
                file.write(f"3 {element[0] - 1} {element[1] - 1}\n")
            elif len(element) == 3:  # For fluid elements which are triangles
                file.write(f"5 {element[0] - 1} {element[1] - 1} {element[2] - 1}\n")
            else:
                raise ValueError(f"Unexpected element length: {len(element)}")

    def isConnectedToVolume(self, surface_element, volume_elements):
        # Check if surface element is part of any volume element
        for volume_element in volume_elements:
            if set(surface_element).issubset(set(volume_element)):
                return True
        return False

    def name(self):
        return 'export_su2'

    def shortHelpString(self):
        return "This algorithm exports a mesh suitable for CFD simulations in .su2 format. " \
               "Object layer is the main layer (e.g. dam), inlet layer is for inlets (air/water/species transport), " \
               "outlet layer is for outlet and mesh layer is the one we created using `Create Mesh Vector`."

    def displayName(self):
        return 'Export to SU2 File'

    def createInstance(self):
        return ExportMeshToSu2Algorithm()
