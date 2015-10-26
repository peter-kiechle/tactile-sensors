#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "Miniball.hpp"
#include "guiRenderer3D.h"
#include "guiMain.h"

#define RADPERDEG 0.0174533

#define checkErrors() printOglError(__FILE__, __LINE__)

guiRenderer3D::guiRenderer3D(FrameManager* fm) : guiRenderer(fm) {
	m_frameManager = fm;
	m_frameProcessor = m_frameManager->getFrameProcessor();
	m_mainGUI = NULL;
	delegateConstructor();
}

guiRenderer3D::guiRenderer3D(FrameManager* fm, guiMain *gui) : guiRenderer(fm, gui) {
	m_frameManager = fm;
	m_frameProcessor = m_frameManager->getFrameProcessor();
	m_mainGUI = gui;
	delegateConstructor();
}

guiRenderer3D::~guiRenderer3D() { }


void guiRenderer3D::delegateConstructor() {
	if(m_frameManager->getSensorInfo().nb_matrices == 0) {
		m_isRendering = false;
	}

	//m_colormap.createColorTable(YELLOW_RED, 3700);
	m_colormap.createColorTable(BREWER_YlOrRd, 3700);

	// Create OpenGL context for on-screen rendering
	Glib::RefPtr<Gdk::GL::Config> GLConfig = Gdk::GL::Config::create(
			Gdk::GL::MODE_RGBA   | // 32 bpp
			Gdk::GL::MODE_DEPTH  | // Z-Buffer
			Gdk::GL::MODE_DOUBLE); // Doublebuffering

	if(!GLConfig)
		throw std::runtime_error("Cannot create OpenGL frame buffer configuration");

	// Accept the configuration
	set_gl_capability(GLConfig);

	// Create OpenGL context and Gdk pixmap for off-screen rendering
	m_GLConfig_Offscreen = Gdk::GL::Config::create(Gdk::GL::MODE_RGB |
			Gdk::GL::MODE_DEPTH |
			Gdk::GL::MODE_SINGLE);

	if (!m_GLConfig_Offscreen)
		throw std::runtime_error("Cannot create OpenGL frame buffer configuration");

	set_colormap(m_GLConfig_Offscreen->get_colormap());

	setOffscreenSize(2048, 2048);


	// Register the fact that we want to receive these events
	add_events(Gdk::BUTTON1_MOTION_MASK);
	add_events(Gdk::BUTTON2_MOTION_MASK);
	add_events(Gdk::BUTTON3_MOTION_MASK);
	add_events(Gdk::BUTTON_PRESS_MASK);
	add_events(Gdk::BUTTON_RELEASE_MASK);
	add_events(Gdk::VISIBILITY_NOTIFY_MASK);
	add_events(Gdk::KEY_PRESS_MASK);
	add_events(Gdk::KEY_RELEASE_MASK);

	// Mouse movement
	m_mouseSensitivity = 0.05;
	m_mouseLeftButtonDown = false;
	m_mouseRightButtonDown = false;
	m_mouseMiddleButtonDown = false;
	m_mouseDragging = false;

	m_mouseLeftButtonDown = false;
	m_selectionMode = false;

	// Free view camera
	m_cameraMoveLeft = false;
	m_cameraMoveRight = false;
	m_cameraMoveUp = false;
	m_cameraMoveDown = false;
	m_cameraMoveForward = false;
	m_cameraMoveBackward = false;
	m_rotationDelta << 0, 0;

	// Camera position and view
	m_pos2D_default = Eigen::Vector3d(0.0, 100.0, 150.0);
	m_view2D_default = Eigen::Vector3d(0.0, 0.0, -1.0);
	m_pos3D_default = Eigen::Vector3d(175.0, 150.0, 175.0);
	m_view3D_default = Eigen::Vector3d(-0.7, -0.2, -0.7);
	m_pos2D = m_pos2D_default;
	m_view2D = m_view2D_default;
	m_pos3D = m_pos3D_default;
	m_view3D = m_view3D_default;

	m_camera = boost::shared_ptr<Camera>(new Camera(m_pos2D, m_view2D)); // Use rotateX() and rotateY() to set pitch and yaw
	m_camera->setup();
	m_cameraDelay = 30.0;
	m_speedFactor = 2.5;
	m_isMoving = false;
	m_isLooping = false;

	m_wireFrameMode = false;
	m_drawNormals = false;

	m_liveMode = false;

	m_mode = MATRICES_2D_BARS;
	m_modeMiniball = MB_TRANSPARENT;
}


void guiRenderer3D::init() {

	// Build 2D coordinates of sensor cells
	std::vector<double> centerX;
	std::vector<double> centerY;

	std::vector<double> dimensionX;
	std::vector<double> dimensionY;

	// Collect sensor layout data
	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
		centerX.push_back(m_frameManager->getMatrixInfo(m).matrix_center_x);
		centerY.push_back(m_frameManager->getMatrixInfo(m).matrix_center_y);
		dimensionX.push_back(m_frameManager->getMatrixInfo(m).cells_x * m_frameManager->getMatrixInfo(m).texel_width);
		dimensionY.push_back(m_frameManager->getMatrixInfo(m).cells_y * m_frameManager->getMatrixInfo(m).texel_height);
	}

	double minX = *std::min_element(centerX.begin(), centerX.end());
	double maxX = *std::max_element(centerX.begin(), centerX.end());
	double minY = *std::min_element(centerY.begin(), centerY.end());
	double maxY = *std::max_element(centerY.begin(), centerY.end());
	double rangeX = maxX-minX;
	double rangeY = maxY-minY;


	// Map cell centers to position in grid
	int nBinsX = 3;
	int nBinsY = 2;
	double shiftX = (minX < 0) ? minX : -minX;
	double shiftY = (minY < 0) ? minY : -minY;

	std::vector<int> gridIndexX;
	std::vector<int> gridIndexY;

	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
		int gridX = floor((nBinsX/rangeX)*(centerX[m]-shiftX)); // Index into Grid
		int gridY = ceil(-(nBinsY/rangeY)*(centerY[m]-shiftY)+nBinsY-1); // Reverse order of y-axis

		// clamp values to range [0, nBins-1]
		gridX = gridX > nBinsX-1 ? nBinsX-1 : gridX;
		gridX = gridX < 0 ? 0 : gridX;
		gridY = gridY > nBinsY-1 ? nBinsY-1 : gridY;
		gridY = gridY < 0 ? 0 : gridY;
		gridIndexX.push_back(gridX);
		gridIndexY.push_back(gridY);
	}


	// Calculate new sensor matrix layout

	// Space between sensor matrices
	double sensorSpacerX = 3*m_frameManager->getMatrixInfo(0).texel_width;
	double sensorSpacerY = 3*m_frameManager->getMatrixInfo(0).texel_height;

	double gridCenterX[nBinsX];
	double gridCenterY[nBinsY];
	double runningDimX = 0;
	double runningDimY = 0;

	for(int x = 0; x < nBinsX; x++) {
		std::vector<double> gridDimensions;
		for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
			if(gridIndexX[m] == x) {
				gridDimensions.push_back(dimensionX[m]);
			}
		}
		double dim = *std::max_element(gridDimensions.begin(), gridDimensions.end()) + 2*sensorSpacerX;
		gridCenterX[x] = runningDimX + dim/2.0;
		runningDimX += dim;
	}

	for(int y = 0; y < nBinsY; y++) {
		std::vector<double> gridDimensions;
		for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
			if(gridIndexY[m] == y) {
				gridDimensions.push_back(dimensionY[m]);
			}
		}
		double dim = *std::max_element(gridDimensions.begin(), gridDimensions.end()) + 2*sensorSpacerY;
		gridCenterY[y] = runningDimY + dim/2.0;
		runningDimY += dim;
	}


	// Calculate sensor cell positions: vector overkill ;-)
	m_matrixCenterX.resize(m_frameManager->getNumMatrices());
	m_matrixCenterY.resize(m_frameManager->getNumMatrices());
	m_topLeftCellCenterX.resize(m_frameManager->getNumMatrices());
	m_topLeftCellCenterY.resize(m_frameManager->getNumMatrices());
	m_cellTopLeftX.resize(m_frameManager->getNumCells());
	m_cellTopLeftY.resize(m_frameManager->getNumCells());
	m_cellBottomRightX.resize(m_frameManager->getNumCells());
	m_cellBottomRightY.resize(m_frameManager->getNumCells());

	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {

		matrixInfo &matrix = m_frameManager->getMatrixInfo(m);

		double cellWidth = matrix.texel_width;
		double cellHeight = matrix.texel_height;

		m_matrixCenterX[m] = gridCenterX[gridIndexX[m]];
		m_matrixCenterY[m] = gridCenterY[gridIndexY[m]];
		m_topLeftCellCenterX[m] = m_matrixCenterX[m] - matrix.cells_x/2.0 * matrix.texel_width  + cellWidth/2;
		m_topLeftCellCenterY[m] = m_matrixCenterY[m] - matrix.cells_y/2.0 * matrix.texel_height + cellHeight/2;

		double cellCenterX = m_topLeftCellCenterX[m];
		double cellCenterY = m_topLeftCellCenterY[m];

		for(uint y = 0; y < matrix.cells_y; y++) {
			cellCenterX = m_topLeftCellCenterX[m];
			for(uint x = 0; x < matrix.cells_x; x++) {
				uint cellID = matrix.texel_offset + y * matrix.cells_x + x;

				m_cellTopLeftX[cellID] = cellCenterX - cellWidth/2;
				m_cellTopLeftY[cellID] = cellCenterY - cellHeight/2;
				m_cellBottomRightX[cellID] = m_cellTopLeftX[cellID] + cellWidth;
				m_cellBottomRightY[cellID] = m_cellTopLeftY[cellID] + cellHeight;

				cellCenterX += matrix.texel_width;
			}
			cellCenterY += matrix.texel_height;
		}
	}

	// Calculate bounding box of matrices
	m_leftMargin = *std::min_element(m_cellTopLeftX.begin(), m_cellTopLeftX.end()) - sensorSpacerX;
	m_rightMargin = *std::max_element(m_cellBottomRightX.begin(), m_cellBottomRightX.end()) + sensorSpacerX;
	m_topMargin = *std::min_element(m_cellTopLeftY.begin(), m_cellTopLeftY.end()) - sensorSpacerY;
	m_bottomMargin = *std::max_element(m_cellBottomRightY.begin(), m_cellBottomRightY.end()) + sensorSpacerY;

	m_sensorWidth = m_rightMargin - m_leftMargin;
	m_sensorHeight = m_bottomMargin - m_topMargin;

	// Shift layout in order to use the same light positions in 2D and 3D mode
	float offset_x = -(m_leftMargin + m_sensorWidth / 2.0);
	float offset_y = 100 + m_sensorHeight / 2.0;

	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
		m_matrixCenterX[m] += offset_x;
		m_matrixCenterY[m] = -m_matrixCenterY[m] + offset_y;
		m_topLeftCellCenterX[m] += offset_x;
		m_topLeftCellCenterY[m] = -m_topLeftCellCenterY[m] + offset_y;


		matrixInfo &matrix = m_frameManager->getMatrixInfo(m);
		for(uint y = 0; y < matrix.cells_y; y++) {
			for(uint x = 0; x < matrix.cells_x; x++) {
				uint cellID = matrix.texel_offset + y * matrix.cells_x + x;
				m_cellTopLeftX[cellID] += offset_x;
				m_cellTopLeftY[cellID] = -m_cellTopLeftY[cellID] + offset_y;
				m_cellBottomRightX[cellID] += offset_x;
				m_cellBottomRightY[cellID] = -m_cellBottomRightY[cellID] + offset_y;
			}
		}
	}

	m_isVisible.resize(m_frameManager->getNumMatrices());
	std::fill(m_isVisible.begin(), m_isVisible.end(), true);
}


/**
 * Initialization (called only once)
 */
void guiRenderer3D::on_realize() {
	Gtk::DrawingArea::on_realize(); // Init DrawingArea first
	Glib::RefPtr<Gdk::GL::Window> glwindow = get_gl_window();

	if(!glwindow->gl_begin(get_gl_context()))
		return;

	// get OpenGL version info
	//    const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
	//    const GLubyte* version = glGetString (GL_VERSION); // version as a string
	//    printf("Renderer: %s\n", renderer);
	//    printf("OpenGL version supported %s\n", version);


	//    glClearColor(0.04314, 0.17255, 0.33333, 1.0); // Dark blue
	//    glClearColor(0.0, 0.61568, 0.87843, 1.0); // Light blue
	glClearColor(0.0, 0.0, 0.0, 1.0); // Black
	//    glClearColor(0.15, 0.15, 0.15, 1.0); // Dark Grey
	//    glClearColor(0.3, 0.3, 0.3, 1.0); // Light Grey
	//    glClearColor(1.0, 1.0, 1.0, 1.0); // White
	glClearDepth(1.0);
	glEnable(GL_DEPTH_TEST);

	// Face culling
	//glFrontFace(GL_CW);
	//glCullFace(GL_BACK);
	//glCullFace(GL_FRONT);
	//glEnable(GL_CULL_FACE);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glEnable(GL_POLYGON_SMOOTH);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_MULTISAMPLE);


	// Light
	GLfloat ambientLight[] =  { 0.0, 0.0, 0.0, 1.0 }; // Define light properties
	GLfloat diffuseLight[] =  { 0.4, 0.4, 0.4, 0.4 };
	GLfloat specularLight[] = { 0.6, 0.6, 0.6, 1.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight); // Load light properties
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);

	glLightfv(GL_LIGHT1, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT1, GL_SPECULAR, specularLight);

	glLightfv(GL_LIGHT2, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT2, GL_SPECULAR, specularLight);

	glLightfv(GL_LIGHT3, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT3, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT3, GL_SPECULAR, specularLight);

	GLfloat globalAmbient[] = { 0.3, 0.3, 0.3, 1.0 };
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, globalAmbient);

	glEnable(GL_LIGHTING); // Enable OpenGL lighting
	glShadeModel(GL_SMOOTH);

	glEnable(GL_LIGHT0); // Enable light source
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHT2);
	glEnable(GL_LIGHT3);

	m_lightPosition0 <<  0.0, 100.0, 0.0, 1.0; // Hand "center"
	m_lightPosition1 << -100.0, 100.0, 100.0, 1.0; // For 2D
	m_lightPosition2 <<  100.0, 100.0, 100.0, 1.0; // For 2D
	m_lightPosition3 <<  0.0, 200.0, -50.0, 1.0;  // Opposing finger

	// Material
	GLfloat ambientMaterial[] =  { 0.3, 0.3, 0.3, 1.0 };
	GLfloat diffuseMaterial[] =  { 0.3, 0.3, 0.3, 1.0 };

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambientMaterial);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseMaterial);

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL); // Let vertex color override material properties

	glwindow->gl_end();
}


/**
 * Called when resizing
 */
bool guiRenderer3D::on_configure_event(GdkEventConfigure* event) {
	Glib::RefPtr<Gdk::GL::Window> glwindow = get_gl_window();

	if(!glwindow->gl_begin(get_gl_context()))
		return false;

	glViewport(0, 0, get_width(), get_height());
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	double aspect = static_cast<double>(get_width()) / static_cast<double>(get_height());
	gluPerspective(45.0, aspect, 0.1, 1000.0);
	glMatrixMode(GL_MODELVIEW);

	glwindow->gl_end();

	return true;
}


/**
 * Draw scene
 */
bool guiRenderer3D::on_expose_event(GdkEventExpose* event) {
	Glib::RefPtr<Gdk::GL::Window> glwindow = get_gl_window();

	if(!glwindow->gl_begin(get_gl_context()))
		return false;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Draw scene
	if(m_isRendering) {
		glLoadIdentity();
		updateCamera();

		glLightfv(GL_LIGHT0, GL_POSITION, m_lightPosition0.data()); // Light position has to be transformed by the model-view matrix
		glLightfv(GL_LIGHT1, GL_POSITION, m_lightPosition1.data());
		glLightfv(GL_LIGHT2, GL_POSITION, m_lightPosition2.data());
		glLightfv(GL_LIGHT3, GL_POSITION, m_lightPosition3.data());

		drawGrid(200.0);
		drawCoordinateAxes(3.0);
		drawMatrices();
	}

	glwindow->swap_buffers(); // Exchange draw buffers (double buffering)

	glwindow->gl_end();
	return true;
}


/**
 * Mouse button pressed
 */
bool guiRenderer3D::on_button_press_event(GdkEventButton* event) {

	// Left mouse button
	if(event->button == 1) {
		m_mouseLeftButtonDown = true;
		m_mouseLeftPressed.x() = event->x; // Initialize the starting coordinates for further mouse movement
		m_mouseLeftPressed.y() = get_height()-event->y; // Invert y-axis

		if(m_selectionMode) {
			determineSelection(m_mouseLeftPressed.x(), m_mouseLeftPressed.y(), 1, 1);
			renderLoop();
		}
	}

	// Right mouse button
	if(event->button == 3) {
		m_mouseRightButtonDown = true;
		m_mouseRightPressed.x() = event->x; // Initialize the starting coordinates for further mouse movement
		m_mouseRightPressed.y() = get_height()-event->y; // Invert y-axis
	}

	return true;
}


/**
 * Mouse button released
 */
bool guiRenderer3D::on_button_release_event(GdkEventButton* event) {
	// Left button
	if(event->button == 1) {
		m_mouseLeftButtonDown = false;
	}

	if(event->button == 3) {
		m_mouseRightButtonDown = false;
	}

	m_mouseDragging = false;
	return true;
}

/**
 * Moving the mouse with pressed buttons
 */
bool guiRenderer3D::on_motion_notify_event(GdkEventMotion* event) {

	if(m_selectionMode) {
		if(m_mouseLeftButtonDown) {
			m_mouseDragging = true;
			determineSelection(event->x, get_height()-event->y, 1, 1); // Invert y-axis
			renderLoop();
		}

	} else {
		//Compute mouse position deltas to previous position
		Eigen::Vector2i delta(0, 0); // Mouse position deltas

		// Rotation (left mouse button)
		if(m_mouseLeftButtonDown) {
			delta.x() = m_mouseLeftPressed.x() - event->x;
			delta.y() = m_mouseLeftPressed.y() - (get_height() - event->y);
			m_mouseLeftPressed.x() = event->x; // Overwrite the coordinate with current x position
			m_mouseLeftPressed.y() = get_height() - event->y;

			m_rotationDelta.x() += delta.y();	// Rotate upwards/downwards
			m_rotationDelta.y() -= delta.x();	// Rotate left/right
			renderLoop();
		}

		// Forward/backward (right mouse button)
		if(m_mouseRightButtonDown) {

			delta.y() = m_mouseRightPressed.y() - (get_height() - event->y);
			m_mouseRightPressed.y() = (get_height() - event->y);

			if(delta.y() < 0) { // Forwards
				m_camera->moveZ(0.25 * m_speedFactor);
			}
			else if(delta.y() > 0) { // Backwards
				m_camera->moveZ(-0.25 * m_speedFactor);
			}

			renderLoop();
		}

	}

	return true;
}

bool guiRenderer3D::on_key_press_event(GdkEventKey* event) {
	bool refreshRenderer = false;

	// Selection Mode
	if(event->keyval == GDK_KEY_Control_L) {
		m_selectionMode = true;
	}

	// Wireframe
	if(event->keyval == GDK_KEY_W || event->keyval == GDK_KEY_w) {
		if(m_wireFrameMode) {
			glPolygonMode( GL_FRONT_AND_BACK, GL_FILL ); // Disable wireframe mode
			glLineWidth(1.0);
		} else {
			glLineWidth(3.0);
			glPolygonMode( GL_FRONT_AND_BACK, GL_LINE ); // Enable wireframe mode
		}
		m_wireFrameMode = !m_wireFrameMode;
		refreshRenderer = true;
	}

	// Restore default camera position
	if(event->keyval == GDK_KEY_R || event->keyval == GDK_KEY_r) {
		std::fill(m_isVisible.begin(), m_isVisible.end(), true);
		if(m_mode == MATRICES_2D_FLAT || m_mode == MATRICES_2D_BARS) { // 2D rendering mode
			m_camera->setPosition(m_pos2D_default);
			m_camera->setView(m_view2D_default);
		} else { // 3D rendering mode
			m_camera->setPosition(m_pos3D_default);
			m_camera->setView(m_view3D_default);
		}
		refreshRenderer = true;
	}

	// Center matrix
	if(m_mode == MATRICES_2D_FLAT || m_mode == MATRICES_2D_BARS) {
		if(event->keyval == GDK_KEY_0 || event->keyval == GDK_KEY_KP_0) {
			std::fill(m_isVisible.begin(), m_isVisible.end(), false);
			m_isVisible[0] = true;
			m_camera->setPosition(Eigen::Vector3d(m_matrixCenterX[0], m_matrixCenterY[0], 80.0)); //65
			m_camera->setView(Eigen::Vector3d(0.0, 0.0, -1.0));
			refreshRenderer = true;
		}
		if(event->keyval == GDK_KEY_1 || event->keyval == GDK_KEY_KP_1) {
			std::fill(m_isVisible.begin(), m_isVisible.end(), false);
			m_isVisible[1] = true;
			m_camera->setPosition(Eigen::Vector3d(m_matrixCenterX[1], m_matrixCenterY[1], 80.0));
			m_camera->setView(Eigen::Vector3d(0.0, 0.0, -1.0));
			refreshRenderer = true;
		}
		if(event->keyval == GDK_KEY_2 || event->keyval == GDK_KEY_KP_2) {
			std::fill(m_isVisible.begin(), m_isVisible.end(), false);
			m_isVisible[2] = true;
			m_camera->setPosition(Eigen::Vector3d(m_matrixCenterX[2], m_matrixCenterY[2], 80.0));
			m_camera->setView(Eigen::Vector3d(0.0, 0.0, -1.0));
			refreshRenderer = true;
		}
		if(event->keyval == GDK_KEY_3 || event->keyval == GDK_KEY_KP_3) {
			std::fill(m_isVisible.begin(), m_isVisible.end(), false);
			m_isVisible[3] = true;
			m_camera->setPosition(Eigen::Vector3d(m_matrixCenterX[3], m_matrixCenterY[3], 80.0));
			m_camera->setView(Eigen::Vector3d(0.0, 0.0, -1.0));
			refreshRenderer = true;
		}
		if(event->keyval == GDK_KEY_4 || event->keyval == GDK_KEY_KP_4) {
			std::fill(m_isVisible.begin(), m_isVisible.end(), false);
			m_isVisible[4] = true;
			m_camera->setPosition(Eigen::Vector3d(m_matrixCenterX[4], m_matrixCenterY[4], 80.0));
			m_camera->setView(Eigen::Vector3d(0.0, 0.0, -1.0));
			refreshRenderer = true;
		}
		if(event->keyval == GDK_KEY_5 || event->keyval == GDK_KEY_KP_5) {
			std::fill(m_isVisible.begin(), m_isVisible.end(), false);
			m_isVisible[5] = true;
			m_camera->setPosition(Eigen::Vector3d(m_matrixCenterX[5], m_matrixCenterY[5], 80.0));
			m_camera->setView(Eigen::Vector3d(0.0, 0.0, -1.0));
			refreshRenderer = true;
		}
	}

	// Matrix rendering mode
	if(event->keyval == GDK_KEY_V || event->keyval == GDK_KEY_v) {
		if(m_mode == MATRICES_2D_FLAT) {
			m_mode = MATRICES_2D_BARS;
		}
		else if(m_mode == MATRICES_2D_BARS) {
			m_mode = MATRICES_3D_CELLS;
			// Store old, load new camera configuration
			m_pos2D = m_camera->getPosition();
			m_view2D = m_camera->getView();
			m_camera->setPosition(m_pos3D);
			m_camera->setView(m_view3D);
		}
		else if(m_mode == MATRICES_3D_CELLS) {
			m_mode = MATRICES_3D_POINTCLOUD;
		}
		else if(m_mode == MATRICES_3D_POINTCLOUD) {
			m_mode = MATRICES_2D_FLAT;
			// Store old, load new camera configuration
			m_pos3D = m_camera->getPosition();
			m_view3D = m_camera->getView();
			m_camera->setPosition(m_pos2D);
			m_camera->setView(m_view2D);
		}
		refreshRenderer = true;
	}

	// Miniball rendering mode
	if(event->keyval == GDK_KEY_B || event->keyval == GDK_KEY_b) {
		if(m_modeMiniball == MB_NONE) {
			m_modeMiniball = MB_OPAQUE;
		}
		else if(m_modeMiniball == MB_OPAQUE) {
			m_modeMiniball = MB_TRANSPARENT;
		}
		else if(m_modeMiniball == MB_TRANSPARENT) {
			m_modeMiniball = MB_WIREFRAME;
		}
		else if(m_modeMiniball == MB_WIREFRAME) {
			m_modeMiniball = MB_NONE;
		}
		refreshRenderer = true;
	}

	// Normals
	if(event->keyval == GDK_KEY_N || event->keyval == GDK_KEY_n) {
		m_drawNormals = !m_drawNormals;
		refreshRenderer = true;
	}


	// Movement
	// x-axis
	if(event->keyval == GDK_KEY_Left) {
		if(!m_cameraMoveLeft) { // Ignore X11 auto key repeat
			m_cameraMoveLeft = true;
			refreshRenderer = true;
		}
	}
	else if(event->keyval == GDK_KEY_Right) {
		if(!m_cameraMoveRight) { // Ignore X11 auto key repeat
			m_cameraMoveRight = true;
			refreshRenderer = true;
		}
	}
	// y-axis
	else if(event->keyval == GDK_KEY_Page_Up) {
		if(!m_cameraMoveUp) { // Ignore X11 auto key repeat
			m_cameraMoveUp = true;
			refreshRenderer = true;
		}

	}
	else if(event->keyval == GDK_KEY_Page_Down) {
		if(!m_cameraMoveDown) { // Ignore X11 auto key repeat
			m_cameraMoveDown = true;
			refreshRenderer = true;
		}
	}
	// z-axis
	else if(event->keyval == GDK_KEY_Up) {
		if(!m_cameraMoveForward) { // Ignore X11 auto key repeat
			m_cameraMoveForward = true;
			refreshRenderer = true;
		}
	}
	else if(event->keyval == GDK_KEY_Down) {
		if(!m_cameraMoveBackward) { // Ignore X11 auto key repeat
			m_cameraMoveBackward = true;
			refreshRenderer = true;
		}
	}

	if(refreshRenderer) {
		m_isMoving = true;
		renderLoop();
	}

	return true; // Prevent further propagation to focused child-widget
}


bool guiRenderer3D::on_key_release_event(GdkEventKey* event) {

	// Selection Mode
	if(event->keyval == GDK_KEY_Control_L) {
		m_selectionMode = false;
	}

	// x-axis (left-right)
	if(event->keyval == GDK_KEY_Left) {
		m_cameraMoveLeft = false;
	}
	else if(event->keyval == GDK_KEY_Right) {
		m_cameraMoveRight = false;
	}

	// y-axis (bottom-top)
	else if(event->keyval == GDK_KEY_Page_Up) {
		m_cameraMoveUp = false;
	}
	else if(event->keyval == GDK_KEY_Page_Down) {
		m_cameraMoveDown = false;
	}

	// z-axis (backward-forward)
	else if(event->keyval == GDK_KEY_Up) {
		m_cameraMoveForward = false;
	}
	else if(event->keyval == GDK_KEY_Down) {
		m_cameraMoveBackward = false;
	}

	if(!(m_cameraMoveForward || m_cameraMoveBackward || m_cameraMoveLeft || m_cameraMoveRight || m_cameraMoveUp || m_cameraMoveDown)) {
		m_isMoving = false;
	}
	return true; // Prevent further propagation to focused child-widget
}

void guiRenderer3D::moveCamera() {
	// Camera translation
	if(m_cameraMoveForward) {
		m_camera->moveZ(m_speedFactor);
	}
	if(m_cameraMoveBackward) {
		m_camera->moveZ(-m_speedFactor);
	}
	if(m_cameraMoveLeft) {
		m_camera->moveX(-m_speedFactor);
	}
	if(m_cameraMoveRight) {
		m_camera->moveX(m_speedFactor);
	}
	if(m_cameraMoveUp) {
		m_camera->moveY(m_speedFactor);
	}
	if(m_cameraMoveDown) {
		m_camera->moveY(-m_speedFactor);
	}
}


void guiRenderer3D::rotateCamera() {
	// Camera rotation
	if(m_rotationDelta.y() != 0) {
		m_camera->rotateY(m_rotationDelta.y() * m_mouseSensitivity); //rotate left/right (around Up-Vector)
		m_rotationDelta.y() = 0;
	}
	if(m_rotationDelta.x() != 0) {
		m_camera->rotateX(m_rotationDelta.x() * m_mouseSensitivity); //rotate up/down (around Right-Vector)
		m_rotationDelta.x() = 0;
	}
}

void guiRenderer3D::updateCamera() {

	// Process pending camera action
	moveCamera();
	rotateCamera();

	// Perform actual change
	m_camera->setup();
}


/**
 * Implements a render loop that periodically updates the scene
 * for smooth camera movements at a constant frame rate using gtkmm's timeout signal
 */
bool guiRenderer3D::renderLoop() {

	if(!m_liveMode) {
		bool requestRendering = false;
		// Set up a timer for smooth movements
		if(!m_isLooping) { // First and only request
			m_isLooping = true;
			m_currentTime = utils::getCurrentTimeMilliseconds(); // Get current system time in milliseconds
			m_targetTime = m_currentTime + m_cameraDelay;
			sigc::slot<bool> slot = sigc::mem_fun(*this, &guiRenderer3D::renderLoop); // Note: Since renderLoop() returns true, this function is called repeatedly
			m_cameraMovementConnection = Glib::signal_timeout().connect(slot, m_cameraDelay); // Call every cameraDelay ms
			// cameraMovementConnection = Glib::signal_idle().connect(slot); // As fast as possible
			requestRendering = true;
		} else { // There is already a request
			m_currentTime = utils::getCurrentTimeMilliseconds();
			if(m_currentTime >= m_targetTime ) { // Limit frame rate
				m_targetTime = m_currentTime + m_cameraDelay;
				requestRendering = true;
			}
		}

		if(requestRendering) {
			this->queue_draw(); // Add repaint request to the GTK+ scheduler to execute the callback on_expose_event()
		}

		// Stop the timer again
		if(!m_isMoving) {
			// Diconnect the signal handler
			m_cameraMovementConnection.disconnect();
			m_isLooping = false;
		}

	}
	return true;
}


void guiRenderer3D::drawMatrices() {
	if(m_mode == MATRICES_2D_FLAT) {
		drawMatrices2DFlat();
	}
	else if(m_mode == MATRICES_2D_BARS) {
		drawMatrices2DBars();
	}
	else if(m_mode == MATRICES_3D_CELLS) {
		drawMatrices3DCells();
	}
	else if(m_mode == MATRICES_3D_POINTCLOUD) {
		drawMatrices3DPointCloud();
	}
}


void guiRenderer3D::drawMatrices2DFlat() {

	double cellMarginX = 0.1;
	double cellMarginY = 0.1;

	glPushMatrix();

	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
		if(m_isVisible[m]) {
			matrixInfo &matrix = m_frameManager->getMatrixInfo(m);
			TSFrame* tsFrame = m_frameManager->getCurrentFilteredFrame();

			float topLeftCellCenterX = m_topLeftCellCenterX[m];
			float topLeftCellCenterY = m_topLeftCellCenterY[m];

			float cellSpanX = matrix.texel_width/2   - cellMarginX;
			float cellSpanY = matrix.texel_height/2  - cellMarginY;

			float cellCenterX = topLeftCellCenterX;
			float cellCenterY = topLeftCellCenterY;

			// Eigen::Map object and tsFrame->cells share the same data in memory
			Eigen::Map<SensorMatrix> originalMatrixMap(tsFrame->cells.data()+matrix.texel_offset, matrix.cells_y, matrix.cells_x);

			for(uint y = 0; y < matrix.cells_y; y++) {
				cellCenterX = topLeftCellCenterX;
				for(uint x = 0; x < matrix.cells_x; x++) {

					bool maskedStatic = m_frameManager->getStaticMask(m, x, y);

					if(maskedStatic) {
						uint cellID = matrix.texel_offset + y * matrix.cells_x + x;
						float value = originalMatrixMap(y,x);

						RGB color;
						bool maskedDynamic = m_frameManager->getDynamicMask(m, x, y);
						if(maskedDynamic) {
							color = determineColor(value);
						} else {
							color = RGB(0.2, 0.2, 0.2); // Light Grey
						}

						if(m_frameManager->isSelected(cellID)) {
							color.r = 0.0;
							color.g = 1.0;
							color.b = 0.0;
						}

						// Draw sensor cell rectangle
						glLoadName(cellID); // Load cell name on the OpenGL name-stack (for picking)
						glColor4f(color.r, color.g, color.b, 1.0);
						glNormal3f(0.0, 0.0, 1.0);
						glBegin(GL_QUADS);
						glVertex3f(cellCenterX - cellSpanX, cellCenterY + cellSpanY, 0.0f ); // Top left of the quad
						glVertex3f(cellCenterX - cellSpanX, cellCenterY - cellSpanY, 0.0f ); // Bottom left of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY - cellSpanY, 0.0f ); // Bottom right of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY + cellSpanY, 0.0f ); // Top right of the quad
						glEnd();

						// Highlight selected cells
						if(m_frameManager->isSelected(cellID)) {
							glColor4f(0.0, 1.0, 0.0, 0.5);
							glBegin(GL_QUADS);
							glVertex3f(cellCenterX - cellSpanX, cellCenterY + cellSpanY, 0.0f ); // Top left of the quad
							glVertex3f(cellCenterX - cellSpanX, cellCenterY - cellSpanY, 0.0f ); // Bottom left of the quad
							glVertex3f(cellCenterX + cellSpanX, cellCenterY - cellSpanY, 0.0f ); // Bottom right of the quad
							glVertex3f(cellCenterX + cellSpanX, cellCenterY + cellSpanY, 0.0f ); // Top right of the quad
							glEnd();
							glColor4f(0.0, 1.0, 0.0, 1.0);
							glLineWidth(2.0);
							glBegin(GL_LINE_LOOP);
							glVertex3f(cellCenterX - cellSpanX, cellCenterY + cellSpanY, 0.0f ); // Top left of the quad
							glVertex3f(cellCenterX - cellSpanX, cellCenterY - cellSpanY, 0.0f ); // Bottom left of the quad
							glVertex3f(cellCenterX + cellSpanX, cellCenterY - cellSpanY, 0.0f ); // Bottom right of the quad
							glVertex3f(cellCenterX + cellSpanX, cellCenterY + cellSpanY, 0.0f ); // Top right of the quad
							glVertex3f(cellCenterX - cellSpanX, cellCenterY + cellSpanY, 0.0f ); // Top left of the quad
							glEnd();
						}
					}
					cellCenterX += matrix.texel_width;
				}
				cellCenterY -= matrix.texel_height;
			}
			checkErrors();
		}
	}
	glPopMatrix();
}


void guiRenderer3D::drawMatrices2DBars() {

	double cellMarginX = 0.2; // 0.15
	double cellMarginY = 0.2; // 0.15
	float scaleFactor = 0.005; // bar height

	glPushMatrix();

	// 3D Bars
	for(uint m = 0; m < m_frameManager->getNumMatrices(); m++) {
		if(m_isVisible[m]) {
			matrixInfo &matrix = m_frameManager->getMatrixInfo(m);
			TSFrame* tsFrame = m_frameManager->getCurrentFilteredFrame();

			float topLeftCellCenterX = m_topLeftCellCenterX[m];
			float topLeftCellCenterY = m_topLeftCellCenterY[m];

			float cellSpanX = matrix.texel_width/2  - cellMarginX;
			float cellSpanY = matrix.texel_height/2 - cellMarginY;

			float cellCenterX = topLeftCellCenterX;
			float cellCenterY = topLeftCellCenterY;


			// Border padding
			glBegin(GL_QUADS);
			glColor4f(0, 0, 0, 1.0);
			double z = -0.1;
			if(m % 2 == 1) { // Fingertips
				// Curved part
				double borderLeft = m_topLeftCellCenterX[m] + matrix.texel_width/2 - cellMarginX;
				double borderRight = borderLeft + 4*matrix.texel_width + 2*cellMarginX;
				double borderTop = m_topLeftCellCenterY[m] + matrix.texel_width/2 + cellMarginX;
				double borderBottom = borderTop - 5*matrix.texel_height;
				glNormal3f(0.0, 0.0, 1.0);
				glVertex3f(borderLeft, borderTop, z);
				glVertex3f(borderLeft, borderBottom, z);
				glVertex3f(borderRight, borderBottom, z);
				glVertex3f(borderRight, borderTop, z);

				// Flat part
				borderLeft = m_topLeftCellCenterX[m] - matrix.texel_width/2 - cellMarginX;
				borderRight = borderLeft + matrix.cells_x * matrix.texel_width + 2*cellMarginX;
				borderTop = borderBottom;
				borderBottom = borderTop - 8*matrix.texel_height - 2*cellMarginY;
				glNormal3f(0.0, 0.0, 1.0);
				glVertex3f(borderLeft, borderTop, z);
				glVertex3f(borderLeft, borderBottom, z);
				glVertex3f(borderRight, borderBottom, z);
				glVertex3f(borderRight, borderTop, z);

			} else {
				double borderLeft = m_topLeftCellCenterX[m] - matrix.texel_width/2 - cellMarginX;
				double borderRight = borderLeft + matrix.cells_x * matrix.texel_width + 2*cellMarginX;
				double borderTop = m_topLeftCellCenterY[m] + matrix.texel_width/2 + cellMarginX;
				double borderBottom = borderTop - matrix.cells_y * matrix.texel_height - 2*cellMarginY;

				glNormal3f(0.0, 0.0, 1.0);
				glVertex3f(borderLeft, borderTop, z);
				glVertex3f(borderLeft, borderBottom, z);
				glVertex3f(borderRight, borderBottom, z);
				glVertex3f(borderRight, borderTop, z);
			}
			glEnd();


			// Eigen::Map object and tsFrame->cells share the same data in memory
			Eigen::Map<SensorMatrix> originalMatrixMap(tsFrame->cells.data()+matrix.texel_offset, matrix.cells_y, matrix.cells_x);

			// Copy tsFrame->cells to Eigen::Matrix
			//SensorMatrix matrixCopy = Map<SensorMatrix>(tsFrame->cells.data()+matrix.texel_offset, matrix.cells_y, matrix.cells_x);

			for(uint y = 0; y < matrix.cells_y; y++) {
				cellCenterX = topLeftCellCenterX;
				for(uint x = 0; x < matrix.cells_x; x++) {

					bool maskedStatic = m_frameManager->getStaticMask(m, x, y);

					if(maskedStatic) {
						uint cellID = matrix.texel_offset + y * matrix.cells_x + x;
						float value = originalMatrixMap(y,x);

						RGB color;
						bool maskedDynamic = m_frameManager->getDynamicMask(m, x, y);
						if(maskedDynamic) {
							color = determineColor(value);
						} else {
							color = RGB(0.2, 0.2, 0.2); // Light Grey
						}

						if(m_frameManager->isSelected(cellID)) {
							color.r = 0.0;
							color.g = 1.0;
							color.b = 0.0;
						}

						// Draw sensor cell rectangle
						glLoadName(cellID); // Load cell name on the OpenGL name-stack (for picking)

						glBegin(GL_QUADS);

						glColor4f(color.r, color.g, color.b, 1.0);

						// Front
						glNormal3f(0.0, 0.0, 1.0);
						glVertex3f(cellCenterX - cellSpanX, cellCenterY + cellSpanY, value*scaleFactor ); // Top left of the quad
						glVertex3f(cellCenterX - cellSpanX, cellCenterY - cellSpanY, value*scaleFactor ); // Bottom left of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY - cellSpanY, value*scaleFactor ); // Bottom right of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY + cellSpanY, value*scaleFactor ); // Top right of the quad

						// Left side
						glNormal3f(-1.0, 0.0, 0.0);
						glVertex3f(cellCenterX - cellSpanX, cellCenterY + cellSpanY, 0.0 ); // Top left of the quad
						glVertex3f(cellCenterX - cellSpanX, cellCenterY - cellSpanY, 0.0 ); // Bottom left of the quad
						glVertex3f(cellCenterX - cellSpanX, cellCenterY - cellSpanY, value*scaleFactor ); // Bottom left of the quad
						glVertex3f(cellCenterX - cellSpanX, cellCenterY + cellSpanY, value*scaleFactor ); // Top left of the quad

						// Right side
						glNormal3f(1.0, 0.0, 0.0);
						glVertex3f(cellCenterX + cellSpanX, cellCenterY - cellSpanY, 0.0 ); // Bottom right of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY + cellSpanY, 0.0 ); // Top right of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY + cellSpanY, value*scaleFactor ); // Top right of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY - cellSpanY, value*scaleFactor ); // Bottom right of the quad

						// Bottom
						glNormal3f(0.0, -1.0, 0.0);
						glVertex3f(cellCenterX - cellSpanX, cellCenterY - cellSpanY, 0.0 ); // Bottom left of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY - cellSpanY, 0.0 ); // Bottom right of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY - cellSpanY, value*scaleFactor ); // Bottom right of the quad
						glVertex3f(cellCenterX - cellSpanX, cellCenterY - cellSpanY, value*scaleFactor ); // Bottom left of the quad

						// Top
						glNormal3f(0.0, 1.0, 0.0);
						glVertex3f(cellCenterX + cellSpanX, cellCenterY + cellSpanY, 0.0 ); // Top right of the quad
						glVertex3f(cellCenterX - cellSpanX, cellCenterY + cellSpanY, 0.0 ); // Top left of the quad
						glVertex3f(cellCenterX - cellSpanX, cellCenterY + cellSpanY, value*scaleFactor ); // Top left of the quad
						glVertex3f(cellCenterX + cellSpanX, cellCenterY + cellSpanY, value*scaleFactor ); // Top right of the quad

						glEnd();

					}
					cellCenterX += matrix.texel_width;
				}
				cellCenterY -= matrix.texel_height; // Note: OpenGL's "reversed" y-axis
			}
			checkErrors();
		}
	}
	glPopMatrix();
}


void guiRenderer3D::drawMatrices3DCells() {

	if(m_frameManager->getJointAngleFrameAvailable()) {

		JointAngleFrame *jointAngleFrame = m_frameManager->getCurrentJointAngleFrame();
		std::vector<double>& angles_deg = jointAngleFrame->angles;

		// Joint angles [phi0 .. phi6]
		std::vector<double> angles_rad(7);
		// Loop unrolling for debugging reasons ;-)
		angles_rad[0] = utils::degToRad(angles_deg[0]); // Rotational axis (Finger 0 + 2)
		angles_rad[1] = utils::degToRad(angles_deg[1]); // Finger 0
		angles_rad[2] = utils::degToRad(angles_deg[2]);
		angles_rad[3] = utils::degToRad(angles_deg[3]); // Finger 1
		angles_rad[4] = utils::degToRad(angles_deg[4]);
		angles_rad[5] = utils::degToRad(angles_deg[5]); // Finger 2
		angles_rad[6] = utils::degToRad(angles_deg[6]);

		m_forwardKinematics.setAngles(angles_rad);

		TSFrame* tsFrame = m_frameManager->getCurrentFilteredFrame();
		RGB color;
		float value;

		// Prepare 3D data points of matrix cells
		std::vector<std::vector<double> > points;
		std::vector<std::vector<double> > activePoints;
		std::vector<int> activeCells(6,0);

		for(uint m = 0; m < 6; m++) {
			matrixInfo &matrixInfo = m_frameManager->getMatrixInfo(m);
			Eigen::Map<SensorMatrix> originalMatrixMap(tsFrame->cells.data()+matrixInfo.texel_offset, matrixInfo.cells_y, matrixInfo.cells_x);
			for(uint y = 0; y < matrixInfo.cells_y; y++) {
				for(uint x = 0; x < matrixInfo.cells_x; x++) {
					value = originalMatrixMap(y,x);
					color = determineColor(value);
					std::vector<double> point = m_forwardKinematics.GetTaxelXYZ(m, x, y);

					if(value > 0.0) {
						activePoints.push_back(point);
						activeCells[m]++;
					}
					point.push_back(color.r);
					point.push_back(color.g);
					point.push_back(color.b);
					points.push_back(point);
				}
			}
		}

		// -----------
		// Draw cells
		// -----------
		float cellSpan = 1.55;

		std::vector<Eigen::Vector3d> quad;

		glPushMatrix();
		glEnable(GL_LIGHTING);


		for(uint m = 0; m < 6; m++) {
			matrixInfo &matrixInfo = m_frameManager->getMatrixInfo(m);
			Eigen::Map<SensorMatrix> originalMatrixMap(tsFrame->cells.data()+matrixInfo.texel_offset, matrixInfo.cells_y, matrixInfo.cells_x);

			for(uint y = 0; y < matrixInfo.cells_y; y++) {
				for(uint x = 0; x < matrixInfo.cells_x; x++) {
					bool maskedStatic = m_frameManager->getStaticMask(m, x, y);
					if(maskedStatic) {
						value = originalMatrixMap(y,x);
						color = determineColor(value);

						quad.clear();
						quad.push_back( Eigen::Vector3d( m_forwardKinematics.GetPointOnSensorPlaneXYZ(m, 3.4*x-cellSpan, 3.4*y+cellSpan).data() ));
						quad.push_back( Eigen::Vector3d( m_forwardKinematics.GetPointOnSensorPlaneXYZ(m, 3.4*x+cellSpan, 3.4*y+cellSpan).data() ));
						quad.push_back( Eigen::Vector3d( m_forwardKinematics.GetPointOnSensorPlaneXYZ(m, 3.4*x+cellSpan, 3.4*y-cellSpan).data() ));
						quad.push_back( Eigen::Vector3d( m_forwardKinematics.GetPointOnSensorPlaneXYZ(m, 3.4*x-cellSpan, 3.4*y-cellSpan).data() ));

						// Compute quad's normal
						Eigen::Vector3d normal(0.0, 0.0, 0.0);
						for(int i=0; i<4; i++) {
							normal += quad[i].cross(quad[(i+1)%4]) ; // cross product
						}
						normal.normalize(); // In-place

						if(m_drawNormals) {
							Eigen::Vector3d scaledNormal = 5.0 * normal;
							glBegin(GL_LINES);
							glColor4f(1.0, 0.0, 0.0, 1.0);
							int cellIdx = matrixInfo.texel_offset + y * matrixInfo.cells_x + x;
							glVertex3d(points[cellIdx][1], points[cellIdx][2], points[cellIdx][0]);
							glVertex3d(points[cellIdx][1]+scaledNormal[1], points[cellIdx][2]+scaledNormal[2], points[cellIdx][0]+scaledNormal[0]);
							glEnd();
						}

						glColor4f(color.r, color.g, color.b, 1.0);
						glNormal3d(normal[1], normal[2], normal[0]);

						glBegin(GL_QUADS);
						for(int i = 0; i < 4; i++) {
							glVertex3d(quad[i][1], quad[i][2], quad[i][0]); // z-axis in hand coordinate space should be up-vector in OpenGL
						}
						glEnd();

						// Collext active cells for miniball
						if(value > 0.0) {
							std::vector<double> point = m_forwardKinematics.GetTaxelXYZ(m, x, y);
							activePoints.push_back(point);
							activeCells[m]++;
						}
					}
				}
			}
		}
		glPopMatrix();

		// -----------------------------------------
		// Compute and draw minimal bounding sphere
		// -----------------------------------------
		int numActiveMatrices = 0;
		for(uint m = 0; m < 6; m++) {
			if(activeCells[m]) {
				numActiveMatrices++;
			}
		}

		// Compute the minimal bounding sphere of active cells
		if(m_modeMiniball != MB_NONE && numActiveMatrices >= 2) {
			drawMinmalBoundingSphere(activePoints);
		}

		checkErrors();
	}

}


void guiRenderer3D::drawMatrices3DPointCloud() {

	if(m_frameManager->getJointAngleFrameAvailable()) {

		JointAngleFrame *jointAngleFrame = m_frameManager->getCurrentJointAngleFrame();
		std::vector<double>& angles_deg = jointAngleFrame->angles;

		// Joint angles [phi0 .. phi6]
		std::vector<double> angles_rad(7);
		// Loop unrolling for debugging reasons ;-)
		angles_rad[0] = utils::degToRad(angles_deg[0]); // Rotational axis (Finger 0 + 2)
		angles_rad[1] = utils::degToRad(angles_deg[1]); // Finger 0
		angles_rad[2] = utils::degToRad(angles_deg[2]);
		angles_rad[3] = utils::degToRad(angles_deg[3]); // Finger 1
		angles_rad[4] = utils::degToRad(angles_deg[4]);
		angles_rad[5] = utils::degToRad(angles_deg[5]); // Finger 2
		angles_rad[6] = utils::degToRad(angles_deg[6]);

		m_forwardKinematics.setAngles(angles_rad);

		TSFrame* tsFrame = m_frameManager->getCurrentFilteredFrame();
		RGB color;
		float value;

		// Prepare 3D data points of active cells
		std::vector<std::vector<double> > points;
		std::vector<std::vector<double> > activePoints;

		std::vector<int> activeCells(6,0);

		for(uint m = 0; m < 6; m++) {
			matrixInfo &matrixInfo = m_frameManager->getMatrixInfo(m);
			Eigen::Map<SensorMatrix> originalMatrixMap(tsFrame->cells.data()+matrixInfo.texel_offset, matrixInfo.cells_y, matrixInfo.cells_x);

			for(uint y = 0; y < matrixInfo.cells_y; y++) {
				for(uint x = 0; x < matrixInfo.cells_x; x++) {
					bool maskedStatic = m_frameManager->getStaticMask(m, x, y);
					if(maskedStatic) {
						value = originalMatrixMap(y,x);
						color = determineColor(value);
						std::vector<double> point = m_forwardKinematics.GetTaxelXYZ(m, x, y);

						if(value > 0.0) {
							activePoints.push_back(point);
							activeCells[m]++;
						}
						point.push_back(color.r);
						point.push_back(color.g);
						point.push_back(color.b);
						points.push_back(point);

					}
				}
			}
		}

		// --------------------------------
		// Minimal bounding sphere
		// --------------------------------
		int numActiveMatrices = 0;
		for(uint m = 0; m < 6; m++) {
			if(activeCells[m]) {
				numActiveMatrices++;
			}
		}

		// ------------
		// Point cloud
		// ------------
		glDisable(GL_LIGHTING);
		glPushMatrix();

		glPointSize(4.0f);
		glBegin(GL_POINTS);
		std::vector<std::vector<double> >::iterator it;
		for(it = points.begin(); it != points.end(); ++it) {
			glColor4d((*it)[3], (*it)[4], (*it)[5], 1.0);
			glVertex3d((*it)[1], (*it)[2], (*it)[0]); // z-axis in hand coordinate space should be up-vector in OpenGL
		}
		glEnd();
		//glDisable(GL_DEPTH_TEST);
		glClearDepth(1.0);
		glPopMatrix();

		//glDisable(GL_DEPTH_TEST);
		glClearDepth(1.0);
		if(m_modeMiniball != MB_NONE && numActiveMatrices >= 2) {
			drawMinmalBoundingSphere(activePoints);
		}

		checkErrors();

	}
}


void guiRenderer3D::drawMinmalBoundingSphere(std::vector<std::vector<double> >& activePoints) {
	typedef std::vector<std::vector<double> >::const_iterator PointIterator;
	typedef std::vector<double>::const_iterator CoordIterator;
	typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MB;
	int d = 3; // Dimension of bounding sphere
	MB mb(d, activePoints.begin(), activePoints.end());

	std::vector<double> center(3);

	// Center of the MB
	const double* c = mb.center();
	for(int i=0; i<d; ++i, ++c) {
		center[i] = *c;
	}

	glPushMatrix();
	glEnable(GL_LIGHTING);
	glTranslatef(center[1], center[2], center[0]);
	glRotatef(90, 1, 0, 0);

	GLUquadricObj *sphere;
	sphere = gluNewQuadric();

	if(m_modeMiniball == MB_TRANSPARENT) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);
	}

	if(m_modeMiniball != MB_WIREFRAME) {
		//glColor4f(0.0, 0.61568, 0.87843, 0.5);
		glColor4f(0.0, 1.0, 1.0, 0.6);
		gluSphere(sphere, sqrt(mb.squared_radius()), 32, 32); // radius, slices and stacks
	}
	else {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glLineWidth(2.0);
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE ); // Enable wireframe mode
		glColor4f(0.0, 1.0, 1.0, 0.6);
		gluSphere(sphere, sqrt(mb.squared_radius()), 32, 32); // radius, slices and stacks
		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL ); // Disable wireframe mode
		glLineWidth(1.0);
		glDisable(GL_BLEND);
	}

	gluDeleteQuadric(sphere);

	if(m_modeMiniball == MB_TRANSPARENT) {
		glDisable(GL_BLEND);
		glDisable(GL_CULL_FACE);
	}
	glPopMatrix();
}


void guiRenderer3D::drawGrid(float scale) {

	glDisable(GL_LIGHTING); // Disable lighting for uniform grid color

	int nLinesX = 15;
	int nLinesZ = 15;
	float lowerX = -scale;
	float upperX =  scale;
	float lowerZ = -scale;
	float upperZ =  scale;
	float Y = -100.0;

	float stepX = fabs(lowerX - upperX) / (nLinesX-1);
	float stepZ = fabs(lowerZ - upperZ) / (nLinesZ-1);

	// Transparent base plane
	glColor4f(0.0, 0.61568, 0.87843, 1.0);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glBegin(GL_QUADS);
	glVertex3f(lowerX-0.2, Y-0.1, lowerZ-0.2);
	glVertex3f(lowerX-0.2, Y-0.1, upperZ+0.2);
	glVertex3f(upperX+0.2, Y-0.1, upperZ+0.2);
	glVertex3f(upperX+0.2, Y-0.1, lowerZ-0.2);
	glEnd();

	// Lines
	glLineWidth(2.0);
	glBegin(GL_LINES);
	for(int z = 0; z < nLinesZ; z++) { // Draw x-axis parallel lines
		glVertex3f(lowerX, Y, lowerZ + z*stepZ);
		glVertex3f(upperX, Y, lowerZ + z*stepZ);
	}
	for(int x = 0; x < nLinesX; x++) { // Draw z-axis parallel lines
		glVertex3f(lowerX + x*stepX, Y, lowerZ);
		glVertex3f(lowerX + x*stepX, Y, upperZ);
	}
	glEnd();

	glLineWidth(1.0);
	glDisable(GL_BLEND);
	glEnable(GL_LIGHTING);
}


void guiRenderer3D::drawLine(double x1, double y1, double z1, double x2, double y2, double z2, double diameter) {
	double x = x2 - x1;
	double y = y2 - y1;
	double z = z2 - z1;
	double length = sqrt(x*x + y*y + z*z);

	GLUquadricObj *quadObj;

	glPushMatrix();

	glTranslated(x1, y1, z1);

	if( (x != 0.0) || (y != 0.0) ) {
		glRotated(atan2(y, x)/RADPERDEG, 0.0, 0.0, 1.0);
		glRotated(atan2(sqrt(x*x+y*y), z)/RADPERDEG, 0.0, 1.0, 0.0);
	} else if (z < 0){
		glRotated(180, 1.0, 0.0, 0.0);
	}

	// Shaft
	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, diameter, diameter, length, 32, 1);
	gluDeleteQuadric(quadObj);

	// Bottom
	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);glTranslatef(0, 0, length);
	gluDisk(quadObj, 0.0, diameter, 32, 1);
	gluDeleteQuadric(quadObj);

	// Top
	glTranslatef(0, 0, -length);
	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, diameter, 32, 1);
	gluDeleteQuadric(quadObj);

	glPopMatrix();
}


// http://stackoverflow.com/questions/19332668/drawing-the-axis-with-its-arrow-using-opengl-in-visual-studio-2010-and-c
// Author: The Quantum Physicist
void guiRenderer3D::drawArrow(double x1, double y1, double z1, double x2, double y2, double z2, double diameter) {
	double x = x2 - x1;
	double y = y2 - y1;
	double z = z2 - z1;
	double length = sqrt(x*x + y*y + z*z);

	double coneHeight = 6*diameter;
	double coneDiameter = 3*diameter;

	GLUquadricObj *quadObj;

	glPushMatrix();

	glTranslated(x1, y1, z1);

	if( (x != 0.0) || (y != 0.0) ) {
		glRotated(atan2(y, x)/RADPERDEG, 0.0, 0.0, 1.0);
		glRotated(atan2(sqrt(x*x+y*y), z)/RADPERDEG, 0.0, 1.0, 0.0);
	} else if (z < 0){
		glRotated(180, 1.0, 0.0, 0.0);
	}

	glTranslatef(0, 0, length-coneHeight);

	// Cone
	quadObj = gluNewQuadric();
	gluQuadricDrawStyle (quadObj, GLU_FILL);
	gluQuadricNormals (quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, coneDiameter, 0.0, coneHeight, 32, 1);
	gluDeleteQuadric(quadObj);

	// Bottom of cone
	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, coneDiameter, 32, 1);
	gluDeleteQuadric(quadObj);

	glTranslatef(0,0,-length+coneHeight);

	// Shaft
	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, diameter, diameter, length-coneHeight, 32, 1);
	gluDeleteQuadric(quadObj);

	// Bottom of shaft
	quadObj = gluNewQuadric ();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, diameter, 32, 1);
	gluDeleteQuadric(quadObj);

	glPopMatrix();
}


void guiRenderer3D::drawCoordinateAxes(double length) {

	// X (red)
	glPushMatrix();
	glTranslatef(0,0,0);
	glColor3d(1.0, 0.0, 0.0);
	drawArrow(0,0,0, 2*length,0,0, 0.3);
	glPopMatrix();

	// Y (green)
	glPushMatrix();
	glTranslatef(0,0,0);
	glColor3d(0.0, 1.0, 0.0);
	drawArrow(0,0,0, 0,2*length,0, 0.3);
	glPopMatrix();

	// Z (blue)
	glPushMatrix();
	glTranslatef(0,0,0);
	glColor3d(0.0, 0.0, 1.0);
	drawArrow(0,0,0, 0,0,2*length, 0.3);
	glPopMatrix();
}




/**
 * Retrieve selected sensor cells using OpenGL's selection buffer for picking
 * NOTE: GL_SELECT is deprecated in OpenGL 3.0
 */
void guiRenderer3D::determineSelection(int x, int y, int deltaX, int deltaY) {
	uint selectedCellsBefore = m_frameManager->getNumSelectedCells();
	glDisable(GL_LIGHTING);

	GLint viewport[4];
	GLuint selectionBuffer[64] = {0};
	GLint hits;

	// Init selection buffer
	glSelectBuffer(64, selectionBuffer);

	// Get viewport
	glGetIntegerv(GL_VIEWPORT, viewport);

	// Switching to selection mode
	glRenderMode(GL_SELECT);

	// Clear the name stack
	glInitNames();

	// Push dummy object
	glPushName(0);

	// Now modify the viewing volume, restricting the drawing area to the cursor's neighborhood
	// And draw the objects onto the screen (again)
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluPickMatrix(x, y, deltaX, deltaY, viewport);
	gluPerspective(45.0, (float)viewport[2]/(float)viewport[3], 0.0001, 1000.0);
	glMatrixMode(GL_MODELVIEW); // Select transformation matrix stack
	drawMatrices();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	// Return to render mode and list picked objects
	hits = glRenderMode(GL_RENDER);
	listSelection(hits, selectionBuffer);

	glMatrixMode(GL_MODELVIEW); // select transformation matrix stack
	glEnable(GL_LIGHTING);

	uint selectedCellsAfter = m_frameManager->getNumSelectedCells();

	if(m_mainGUI) {
		if(selectedCellsBefore != selectedCellsAfter) {
			m_mainGUI->updateDataset();
		}
	}
}


void guiRenderer3D::listSelection(GLint hits, GLuint *names) {

	hits = 1; // Only select first/nearest object

	for (int i = 0; i < hits; i++) {
		// names[i * 4 + 3] : Number of hits
		// names[i * 4 + 3] : Min Z
		// names[i * 4 + 3] : Max Z
		// names[i * 4 + 3] : ID
		int id = names[i * 4 + 3];

		// Toggle selection
		if(m_mouseDragging) { // Mark new cell with the same value as the previous one
			m_frameManager->selectCell(id, m_previousSelectionState);
		} else { // Single click (toggle state)
			m_frameManager->isSelected(id) ? m_frameManager->selectCell(id, false) : m_frameManager->selectCell(id, true);
			m_previousSelectionState = m_frameManager->isSelected(id);
			TSFrame* tsFrame = m_frameManager->getCurrentFrame();
			printf("Renderer 3d: Cell %d: %f\n", id, tsFrame->cells[id]);
		}
	}

}



// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.
int guiRenderer3D::printOglError(const char *file, int line) {
	GLenum glErr;
	int retCode = 0;

	glErr = glGetError();
	if (glErr != GL_NO_ERROR) {
		printf("glError in file %s @ line %d: %s\n", file, line, gluErrorString(glErr));
		retCode = 1;
	}
	return retCode;
}


// Invalidate whole window
// Force widget to be redrawn in the near future
void guiRenderer3D::invalidate() {
	Glib::RefPtr<Gdk::Window> window = get_window();
	if(window) {
		get_window()->invalidate_rect(get_allocation(), false);
	}
}

// Update window synchronously (fast)
// Causes redrawing to be done immediately
void guiRenderer3D::update() {
	Glib::RefPtr<Gdk::Window> window = get_window();
	if(window) {
		get_window()->process_updates(false);
	}
}


void guiRenderer3D::renderFrame() {
	this->queue_draw();
}


void guiRenderer3D::renderFrame(uint frameID) {
	m_frameManager->setCurrentFrameID(frameID);
	this->queue_draw();
}


void guiRenderer3D::initOffscreen() {

	//glClearColor(1.0, 1.0, 1.0, 1.0); // White
	glClearColor(0.0, 0.0, 0.0, 1.0); // Black
	glClearDepth(1.0);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glEnable(GL_POLYGON_SMOOTH);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_MULTISAMPLE);

	// Light
	GLfloat ambientLight[] =  { 0.0, 0.0, 0.0, 1.0 }; // Define light properties
	GLfloat diffuseLight[] =  { 0.4, 0.4, 0.4, 0.4 };
	GLfloat specularLight[] = { 0.6, 0.6, 0.6, 1.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight); // Load light properties
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);

	glLightfv(GL_LIGHT1, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT1, GL_SPECULAR, specularLight);

	glLightfv(GL_LIGHT2, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT2, GL_SPECULAR, specularLight);

	glLightfv(GL_LIGHT3, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT3, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT3, GL_SPECULAR, specularLight);

	GLfloat globalAmbient[] = { 0.3, 0.3, 0.3, 1.0 };
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, globalAmbient);

	glEnable(GL_LIGHTING); // Enable OpenGL lighting
	glShadeModel(GL_SMOOTH);

	glEnable(GL_LIGHT0); // Enable light source
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHT2);
	glEnable(GL_LIGHT3);

	// Material
	GLfloat ambientMaterial[] =  { 0.3, 0.3, 0.3, 1.0 };
	GLfloat diffuseMaterial[] =  { 0.3, 0.3, 0.3, 1.0 };

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambientMaterial);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseMaterial);

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL); // Let vertex color override material properties

	// Setup view frustum
	glViewport(0, 0, m_widthOffscreen, m_heightOffscreen);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	double aspect = static_cast<double>(m_widthOffscreen) / static_cast<double>(m_heightOffscreen);
	gluPerspective(45.0, aspect, 0.1, 1000.0);
	glMatrixMode(GL_MODELVIEW);
}


void guiRenderer3D::setOffscreenSize(int width, int height) {
	m_widthOffscreen = width;
	m_heightOffscreen = height;
	m_Pixmap_Offscreen.reset();
	m_Pixmap_Offscreen = Gdk::Pixmap::create(get_window(), m_widthOffscreen, m_heightOffscreen, m_GLConfig_Offscreen->get_depth());
}


void guiRenderer3D::takeScreenshot(int width, int height, std::string filename) {
	if(m_widthOffscreen != width || m_heightOffscreen != height) {
		setOffscreenSize(width, height);
	}
	takeScreenshot(filename);
}


void guiRenderer3D::takeScreenshot(std::string filename) {

	// Set OpenGL-capability to the pixmap (invoke extension method).
	Glib::RefPtr<Gdk::GL::Pixmap> glpixmap = Gdk::GL::ext(m_Pixmap_Offscreen).set_gl_capability(m_GLConfig_Offscreen);

	// Create OpenGL rendering context (not direct rendering).
	Glib::RefPtr<Gdk::GL::Context> m_GLContext_Offscreen = Gdk::GL::Context::create(glpixmap, false);

	if(!glpixmap->gl_begin(m_GLContext_Offscreen))
		return;

	initOffscreen();


	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Draw scene
	updateCamera();

	glLightfv(GL_LIGHT0, GL_POSITION, m_lightPosition0.data()); // Light position has to be transformed by the model-view matrix
	glLightfv(GL_LIGHT1, GL_POSITION, m_lightPosition1.data());
	glLightfv(GL_LIGHT2, GL_POSITION, m_lightPosition2.data());
	glLightfv(GL_LIGHT3, GL_POSITION, m_lightPosition3.data());

	//drawGrid(200.0);
	drawCoordinateAxes(3.0);
	drawMatrices();

	// Read pixels from OpenGL frame buffer
	cv::Mat renderedImage(m_heightOffscreen, m_widthOffscreen, CV_8UC3);
	glReadPixels(0, 0, m_widthOffscreen, m_heightOffscreen, GL_BGR, GL_UNSIGNED_BYTE, (uchar*)renderedImage.data);
	cv::flip(renderedImage, renderedImage, 0); // Origin is bottom left in OpenGL and top left in OpenCV
	imwrite(filename, renderedImage);

	glpixmap->gl_end();
}
