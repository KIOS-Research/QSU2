# -*- coding: utf-8 -*-
"""
/***************************************************************************
 qgis2su2
                                 A QGIS plugin
 This plugin is designed to...
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2023-12-19
        copyright            : (C) 2023 by KIOS CoE
        email                : hassan.syed@ucy.ac.cy
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""

__author__ = 'KIOS CoE'
__date__ = '2023-12-19'
__copyright__ = '(C) 2023 by KIOS CoE'


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load qgis2su2 class from file qgis2su2.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .qgis2su2 import qgis2su2Plugin
    return qgis2su2Plugin()
