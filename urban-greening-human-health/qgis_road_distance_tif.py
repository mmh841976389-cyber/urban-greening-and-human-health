from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterCrs,
                       QgsProcessingException,
                       QgsVectorLayer,
                       QgsCoordinateReferenceSystem,
                       QgsRectangle,
                       QgsCoordinateTransform)
from qgis import processing
import os
import tempfile

class RoadDistanceCalculator(QgsProcessingAlgorithm):
    """
    Compute the distance (in metres) from each cell to the nearest main road.
    """

    # Define algorithm parameters
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    EXTENT = 'EXTENT'
    RESOLUTION = 'RESOLUTION'
    CRS = 'CRS'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return RoadDistanceCalculator()

    def name(self):
        return 'road_distance_calculator'

    def displayName(self):
        return self.tr('Road Distance Calculator')

    def group(self):
        return self.tr('Custom Scripts')

    def groupId(self):
        return 'custom_scripts'

    def shortHelpString(self):
        return self.tr("""
        Generate a raster of the distance (in metres) from each cell in Hangzhou
        to the nearest main road.

        Parameters:
        1. Road layer: input road line layer
        2. Output raster: path of the generated TIFF file
        3. Processing extent: Hangzhou extent (default 118.3,120.72,29.2,30.6)
        4. Resolution: cell size of the output raster (metres)
        5. CRS: projected CRS used for computation (recommended UTM 50N, EPSG:32650)

        Note: processing a large extent requires sufficient memory.
        """)

    def initAlgorithm(self, config=None):
        # Add input parameters
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT,
                self.tr('Road layer'),
                [QgsProcessing.TypeVectorLine],
                optional=False
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr('Output raster'),
                optional=False
            )
        )

        self.addParameter(
            QgsProcessingParameterExtent(
                self.EXTENT,
                self.tr('Processing extent'),
                defaultValue='118.3,120.72,29.2,30.6'  # Hangzhou extent
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.RESOLUTION,
                self.tr('Resolution (m)'),
                type=QgsProcessingParameterNumber.Integer,
                minValue=10,
                maxValue=1000,
                defaultValue=100
            )
        )

        self.addParameter(
            QgsProcessingParameterCrs(
                self.CRS,
                self.tr('Projection CRS'),
                defaultValue='EPSG:32650'  # UTM 50N (suitable for Hangzhou)
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Get input parameters
        road_layer = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        extent = self.parameterAsExtent(parameters, self.EXTENT, context, road_layer.crs())  # Key change: use the layer CRS
        resolution = self.parameterAsInt(parameters, self.RESOLUTION, context)
        target_crs = self.parameterAsCrs(parameters, self.CRS, context)

        # Validate input
        if not road_layer:
            raise QgsProcessingException(self.tr('Invalid road layer input'))

        feedback.pushInfo(f"Start processing: {road_layer.name()}")
        feedback.pushInfo(f"Processing extent: {extent.toString()}")
        feedback.pushInfo(f"Resolution: {resolution} m")
        feedback.pushInfo(f"Target CRS: {target_crs.authid()}")

        # Step 1: reproject the road layer to the target CRS
        feedback.pushInfo("Reprojecting road layer...")

        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_vector_path = os.path.join(temp_dir, "reprojected_roads.shp")

        reproject_params = {
            'INPUT': road_layer,
            'TARGET_CRS': target_crs,
            'OUTPUT': temp_vector_path
        }

        processing.run(
            "native:reprojectlayer",
            reproject_params,
            context=context,
            feedback=feedback
        )

        # Validate reprojection result
        if not os.path.exists(temp_vector_path):
            raise QgsProcessingException(f"Reprojection failed, file not created: {temp_vector_path}")

        # Load the reprojected layer
        reprojected_roads = QgsVectorLayer(temp_vector_path, "ReprojectedRoads", "ogr")
        if not reprojected_roads.isValid():
            raise QgsProcessingException(f"Cannot load the reprojected layer: {temp_vector_path}")

        # Key change: transform the processing extent to the target CRS
        src_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        xform = QgsCoordinateTransform(src_crs, target_crs, context.transformContext())
        projected_extent = xform.transformBoundingBox(extent)
        projected_extent_str = (f"{projected_extent.xMinimum()},{projected_extent.xMaximum()},"
                                f"{projected_extent.yMinimum()},{projected_extent.yMaximum()}")

        feedback.pushInfo(f"Projected processing extent: {projected_extent_str}")

        # Step 2: rasterise the road layer (in the projected CRS)
        feedback.pushInfo("Rasterising road layer...")
        temp_raster_path = os.path.join(temp_dir, "rasterized_roads.tif")

        rasterize_params = {
            'INPUT': reprojected_roads,
            'FIELD': '',
            'BURN': 1,
            'UNITS': 0,  # map units (metres)
            'WIDTH': resolution,
            'HEIGHT': resolution,
            'EXTENT': projected_extent_str,  # use the projected extent
            'NODATA': 0,
            'OPTIONS': '',
            'DATA_TYPE': 0,  # Byte
            'INIT': 0,
            'INVERT': False,
            'OUTPUT': temp_raster_path
        }

        processing.run(
            "gdal:rasterize",
            rasterize_params,
            context=context,
            feedback=feedback
        )

        # Validate rasterisation result
        if not os.path.exists(temp_raster_path):
            raise QgsProcessingException(f"Rasterisation failed, file not created: {temp_raster_path}")

        # Step 3: compute distance (in the projected CRS)
        feedback.pushInfo("Computing road distance...")
        distance_params = {
            'INPUT': temp_raster_path,
            'BAND': 1,
            'VALUES': '1',  # compute distance to value 1 (road)
            'UNITS': 0,     # 0 = map units (metres)
            'MAX_DISTANCE': 0,
            'REPLACE': 0,
            'NODATA': -9999,
            'OPTIONS': '',
            'EXTRA': '',
            'DATA_TYPE': 5,  # Float32
            'OUTPUT': output_path
        }

        processing.run(
            "gdal:proximity",
            distance_params,
            context=context,
            feedback=feedback
        )

        feedback.pushInfo(f"Processing complete! Result saved to: {output_path}")

        # Clean up temporary files
        feedback.pushInfo("Cleaning up temporary files...")
        try:
            # Close the layer to release the file handle
            if reprojected_roads.isValid():
                del reprojected_roads

            # Delete temporary files
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.qpj']:
                file_path = os.path.splitext(temp_vector_path)[0] + ext
                if os.path.exists(file_path):
                    os.remove(file_path)
                    feedback.pushInfo(f"Deleted temporary file: {file_path}")

            if os.path.exists(temp_raster_path):
                os.remove(temp_raster_path)
                feedback.pushInfo(f"Deleted temporary file: {temp_raster_path}")
        except Exception as e:
            feedback.pushWarning(f"Error while cleaning up temporary files: {str(e)}")

        return {self.OUTPUT: output_path}

# Register the algorithm
def register_algorithms():
    from qgis.core import QgsApplication
    QgsApplication.processingRegistry().addAlgorithm(RoadDistanceCalculator())

# Register the algorithm when the script is imported
try:
    from qgis.utils import iface
    register_algorithms()
    iface.messageBar().pushInfo("Success", "Road Distance Calculator added to the Processing toolbox")
except:
    pass
