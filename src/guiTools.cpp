#include "guiTools.h"
#include "guiMain.h"

guiTools::guiTools(Controller *c, guiMain *gui)
:
m_Adjustment_Median_Radius(1.0, 1.0, 2.0, 1.0, 0.0, 0.0),
m_SpinButton_Median_Radius(m_Adjustment_Median_Radius),
m_Adjustment_Gauss_Radius(1.0, 1.0, 10.0, 1.0, 0.0, 0.0),
m_SpinButton_Gauss_Radius(m_Adjustment_Gauss_Radius),
m_Adjustment_Gauss_Sigma(1.0, 0.1, 5.0, 0.1, 0.0, 0.0),
m_HScale_Gauss_Sigma(m_Adjustment_Gauss_Sigma),
m_Adjustment_Bilateral_Radius(1.0, 1.0, 10.0, 1.0, 0.0, 0.0),
m_SpinButton_Bilateral_Radius(m_Adjustment_Bilateral_Radius),
m_Adjustment_Bilateral_Sigma_Color(1.0, 0.1, 500.0, 0.1, 0.0, 0.0),
m_HScale_Bilateral_Sigma_Color(m_Adjustment_Bilateral_Sigma_Color),
m_Adjustment_Bilateral_Sigma_Space(1.0, 0.1, 500.0, 0.1, 0.0, 0.0),
m_HScale_Bilateral_Sigma_Space(m_Adjustment_Bilateral_Sigma_Space),
m_Adjustment_Morphological_Radius(1.0, 1.0, 10.0, 1.0, 0.0, 0.0),
m_SpinButton_Morphological_Radius(m_Adjustment_Morphological_Radius)
{

	controller = c;
	frameManager = controller->getFrameManager();
	frameProcessor = frameManager->getFrameProcessor();
	mainGUI = gui;

	// Median filter
	m_medianFiter = false;
	m_Frame_Median.set_label("2D Median Filter");
	m_CheckButton_Median.set_label("Enable");
	m_CheckButton_Median.signal_clicked().connect( sigc::mem_fun(*this, &guiTools::on_checkbutton_median_clicked) );

	m_CheckButton_Median_Masked.set_label("Use mask");
	m_CheckButton_Median_Masked.set_active(true);
	m_CheckButton_Median_Masked.signal_clicked().connect( sigc::mem_fun(*this, &guiTools::on_checkbutton_median_masked_clicked) );

	m_Label_Median_Radius.set_label("Radius:");
	m_Label_Median_Radius.set_alignment(Gtk::ALIGN_LEFT, Gtk::ALIGN_TOP);

	m_kernelRadiusMedian = 1;
	m_HBox_Median_Radius.pack_start(m_Label_Median_Radius, Gtk::PACK_SHRINK, 5);
	m_Adjustment_Median_Radius.signal_value_changed().connect( sigc::mem_fun(*this, &guiTools::on_spinbutton_median_radius_value_changed) );
	m_HBox_Median_Radius.pack_start(m_SpinButton_Median_Radius, Gtk::PACK_EXPAND_PADDING, 5);

	m_VBox_Median.pack_start(m_CheckButton_Median, Gtk::PACK_START, 5);
	m_VBox_Median.pack_start(m_CheckButton_Median_Masked, Gtk::PACK_SHRINK, 5);
	m_VBox_Median.pack_start(m_HBox_Median_Radius, Gtk::PACK_SHRINK, 5);
	m_VBox_Median.set_border_width(5);

	m_Frame_Median.add(m_VBox_Median);
	m_Frame_Median.set_border_width(5);


	// 3D Median filter
	m_medianFiter3D = false;
	m_Frame_Median3D.set_label("3D Median Filter");
	m_CheckButton_Median3D.set_label("Enable");
	m_CheckButton_Median3D.signal_clicked().connect( sigc::mem_fun(*this, &guiTools::on_checkbutton_median3D_clicked) );

	m_CheckButton_Median3D_Masked.set_label("Use mask");
	m_CheckButton_Median3D_Masked.set_active(true);
	m_CheckButton_Median3D_Masked.signal_clicked().connect( sigc::mem_fun(*this, &guiTools::on_checkbutton_median3D_masked_clicked) );

	m_VBox_Median3D.pack_start(m_CheckButton_Median3D, Gtk::PACK_START, 5);
	m_VBox_Median3D.pack_start(m_CheckButton_Median3D_Masked, Gtk::PACK_SHRINK, 5);
	m_VBox_Median3D.set_border_width(5);

	m_Frame_Median3D.add(m_VBox_Median3D);
	m_Frame_Median3D.set_border_width(5);


	// Gaussian filter
	m_gaussianFiter = false;
	m_Frame_Gaussian.set_label("Gaussian Filter");
	m_CheckButton_Gauss.set_label("Enable");
	m_CheckButton_Gauss.signal_clicked().connect( sigc::mem_fun(*this, &guiTools::on_checkbutton_gauss_clicked) );

	m_Label_Gauss_Radius.set_label("Radius:");
	m_Label_Gauss_Radius.set_alignment(Gtk::ALIGN_LEFT, Gtk::ALIGN_TOP);

	m_CheckButton_Gauss_Auto_Sigma.set_label("Manual Sigma Adjustment");
	m_CheckButton_Gauss_Auto_Sigma.signal_clicked().connect( sigc::mem_fun(*this, &guiTools::on_checkbutton_gauss_auto_sigma_clicked) );

	m_kernelRadiusGauss = 1;
	m_HBox_Gauss_Radius.pack_start(m_Label_Gauss_Radius, Gtk::PACK_SHRINK, 5);
	m_Adjustment_Gauss_Radius.signal_value_changed().connect( sigc::mem_fun(*this, &guiTools::on_spinbutton_gauss_radius_value_changed) );
	m_HBox_Gauss_Radius.pack_start(m_SpinButton_Gauss_Radius, Gtk::PACK_EXPAND_PADDING, 5);

	m_Label_Gauss_Auto_Sigma.set_label("Sigma:");
	m_Label_Gauss_Auto_Sigma.set_alignment(Gtk::ALIGN_LEFT, Gtk::ALIGN_TOP);
	m_Label_Gauss_Auto_Sigma.set_sensitive(false);
	m_HBox_Gauss_Auto_Sigma.pack_start(m_Label_Gauss_Auto_Sigma, Gtk::PACK_SHRINK, 5);
	m_sigmaGauss = -1.0;
	m_Adjustment_Gauss_Sigma.set_value(frameProcessor->calcGaussianSigma(m_kernelRadiusGauss));
	m_HScale_Gauss_Sigma.set_digits(2);
	m_HScale_Gauss_Sigma.set_draw_value(true); // Show position label
	m_HScale_Gauss_Sigma.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_HScale_Gauss_Sigma.set_sensitive(false);
	m_HScale_Gauss_Sigma.signal_button_press_event().connect(sigc::mem_fun(*this, &guiTools::on_slider_gauss_sigma_clicked), false);
	m_HScale_Gauss_Sigma.signal_button_release_event().connect(sigc::mem_fun(*this, &guiTools::on_slider_gauss_sigma_released), false);
	m_HScale_Gauss_Sigma.signal_change_value().connect(sigc::mem_fun(*this, &guiTools::on_slider_gauss_sigma_change_value) );
	m_HBox_Gauss_Auto_Sigma.add(m_HScale_Gauss_Sigma);

	m_VBox_Gauss.pack_start(m_CheckButton_Gauss, Gtk::PACK_START, 5);
	m_VBox_Gauss.pack_start(m_CheckButton_Gauss_Auto_Sigma, Gtk::PACK_SHRINK, 5);
	m_VBox_Gauss.pack_start(m_HBox_Gauss_Radius, Gtk::PACK_SHRINK, 5);
	m_VBox_Gauss.pack_start(m_HBox_Gauss_Auto_Sigma, Gtk::PACK_SHRINK, 5);
	m_VBox_Gauss.set_border_width(5);

	m_Frame_Gaussian.add(m_VBox_Gauss);
	m_Frame_Gaussian.set_border_width(5);


	// Bilateral filter
	m_bilateralFilter = false;
	m_Frame_Bilateral.set_label("Bilateral Filter");
	m_CheckButton_Bilateral.set_label("Enable");
	m_CheckButton_Bilateral.signal_clicked().connect( sigc::mem_fun(*this, &guiTools::on_checkbutton_bilateral_clicked) );

	m_Label_Bilateral_Radius.set_label("Radius:");
	m_Label_Bilateral_Radius.set_alignment(Gtk::ALIGN_LEFT, Gtk::ALIGN_TOP);
	m_kernelRadiusBilateral = 1;
	m_HBox_Bilateral_Radius.pack_start(m_Label_Bilateral_Radius, Gtk::PACK_SHRINK, 5);
	m_Adjustment_Bilateral_Radius.signal_value_changed().connect( sigc::mem_fun(*this, &guiTools::on_spinbutton_bilateral_radius_value_changed) );
	m_HBox_Bilateral_Radius.pack_start(m_SpinButton_Bilateral_Radius, Gtk::PACK_EXPAND_PADDING, 5);

	m_Label_Bilateral_Sigma_Color.set_label("Sigma Color:");
	m_Label_Bilateral_Sigma_Color.set_alignment(Gtk::ALIGN_LEFT, Gtk::ALIGN_TOP);
	m_HBox_Bilateral_Sigma_Color.pack_start(m_Label_Bilateral_Sigma_Color, Gtk::PACK_SHRINK, 5);
	m_sigmaBilateralColor = 50.0;
	m_Adjustment_Bilateral_Sigma_Color.set_value(m_sigmaBilateralColor);
	m_HScale_Bilateral_Sigma_Color.set_digits(2);
	m_HScale_Bilateral_Sigma_Color.set_draw_value(true); // Show position label
	m_HScale_Bilateral_Sigma_Color.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_HScale_Bilateral_Sigma_Color.set_sensitive(true);
	m_HScale_Bilateral_Sigma_Color.signal_button_press_event().connect(sigc::mem_fun(*this, &guiTools::on_slider_bilateral_sigma_color_clicked), false);
	m_HScale_Bilateral_Sigma_Color.signal_button_release_event().connect(sigc::mem_fun(*this, &guiTools::on_slider_bilateral_sigma_color_released), false);
	m_HScale_Bilateral_Sigma_Color.signal_change_value().connect(sigc::mem_fun(*this, &guiTools::on_slider_bilateral_sigma_color_change_value) );
	m_HBox_Bilateral_Sigma_Color.add(m_HScale_Bilateral_Sigma_Color);

	m_Label_Bilateral_Sigma_Space.set_label("Sigma Space:");
	m_Label_Bilateral_Sigma_Space.set_alignment(Gtk::ALIGN_LEFT, Gtk::ALIGN_TOP);
	m_HBox_Bilateral_Sigma_Space.pack_start(m_Label_Bilateral_Sigma_Space, Gtk::PACK_SHRINK, 5);
	m_sigmaBilateralSpace = 3.0;
	m_Adjustment_Bilateral_Sigma_Space.set_value(m_sigmaBilateralSpace);
	m_HScale_Bilateral_Sigma_Space.set_digits(2);
	m_HScale_Bilateral_Sigma_Space.set_draw_value(true); // Show position label
	m_HScale_Bilateral_Sigma_Space.set_value_pos(Gtk::POS_BOTTOM); // Where to draw the position label (if drawn at all)
	m_HScale_Bilateral_Sigma_Space.set_sensitive(true);
	m_HScale_Bilateral_Sigma_Space.signal_button_press_event().connect(sigc::mem_fun(*this, &guiTools::on_slider_bilateral_sigma_space_clicked), false);
	m_HScale_Bilateral_Sigma_Space.signal_button_release_event().connect(sigc::mem_fun(*this, &guiTools::on_slider_bilateral_sigma_space_released), false);
	m_HScale_Bilateral_Sigma_Space.signal_change_value().connect(sigc::mem_fun(*this, &guiTools::on_slider_bilateral_sigma_space_change_value) );
	m_HBox_Bilateral_Sigma_Space.add(m_HScale_Bilateral_Sigma_Space);

	m_VBox_Bilateral.pack_start(m_CheckButton_Bilateral, Gtk::PACK_START, 5);
	m_VBox_Bilateral.pack_start(m_HBox_Bilateral_Radius, Gtk::PACK_SHRINK, 5);
	m_VBox_Bilateral.pack_start(m_HBox_Bilateral_Sigma_Color, Gtk::PACK_SHRINK, 5);
	m_VBox_Bilateral.pack_start(m_HBox_Bilateral_Sigma_Space, Gtk::PACK_SHRINK, 5);
	m_VBox_Bilateral.set_border_width(5);

	m_Frame_Bilateral.add(m_VBox_Bilateral);
	m_Frame_Bilateral.set_border_width(5);


	// Morphological transformation
	m_morphologicalFilter = false;
	m_kernelRadiusMorphological = 1;
	m_kernelTypeMorphological = 0;

	m_Frame_Morphological.set_label("Opening Filter");
	m_CheckButton_Morphological.set_label("Enable");
	m_CheckButton_Morphological.signal_clicked().connect( sigc::mem_fun(*this, &guiTools::on_checkbutton_morphological_clicked) );

	m_CheckButton_Morphological_Masked.set_label("Use mask");
	m_CheckButton_Morphological_Masked.set_active(true);
	m_CheckButton_Morphological_Masked.signal_clicked().connect( sigc::mem_fun(*this, &guiTools::on_checkbutton_morphological_masked_clicked) );


	m_Label_Morphological_Radius.set_label("Radius:");
	m_Label_Morphological_Radius.set_alignment(Gtk::ALIGN_LEFT, Gtk::ALIGN_TOP);
	m_HBox_Morphological_Radius.pack_start(m_Label_Morphological_Radius, Gtk::PACK_SHRINK, 5);
	m_Adjustment_Morphological_Radius.signal_value_changed().connect( sigc::mem_fun(*this, &guiTools::on_spinbutton_morphological_radius_value_changed) );
	m_HBox_Morphological_Radius.pack_start(m_SpinButton_Morphological_Radius, Gtk::PACK_EXPAND_PADDING, 5);

	m_Label_Morphological_Kerneltype.set_label("Kernel type:");
	m_Label_Morphological_Kerneltype.set_alignment(Gtk::ALIGN_LEFT, Gtk::ALIGN_TOP);
	m_HBox_Morphological_Kerneltype.pack_start(m_Label_Morphological_Kerneltype, Gtk::PACK_SHRINK, 5);

	m_Combo_Morphological_Kerneltype.append("Cross-shaped");
	m_Combo_Morphological_Kerneltype.append("Rectangular");
	m_Combo_Morphological_Kerneltype.append("Elliptic");
	m_Combo_Morphological_Kerneltype.set_active(0); // Initial value
	m_Combo_Morphological_Kerneltype.set_size_request(120);
	m_Combo_Morphological_Kerneltype.signal_changed().connect(sigc::mem_fun(*this, &guiTools::on_combo_morphological_kerneltype_changed));
	m_HBox_Morphological_Kerneltype.pack_start(m_Combo_Morphological_Kerneltype, Gtk::PACK_EXPAND_PADDING, 5);

	m_VBox_Morphological.pack_start(m_CheckButton_Morphological, Gtk::PACK_START, 5);
	m_VBox_Morphological.pack_start(m_CheckButton_Morphological_Masked, Gtk::PACK_SHRINK, 5);
	m_VBox_Morphological.pack_start(m_HBox_Morphological_Radius, Gtk::PACK_SHRINK, 5);
	m_VBox_Morphological.pack_start(m_HBox_Morphological_Kerneltype, Gtk::PACK_SHRINK, 5);
	m_VBox_Morphological.set_border_width(5);

	m_Frame_Morphological.add(m_VBox_Morphological);
	m_Frame_Morphological.set_border_width(5);

	m_VBox_Spatial_Filters.pack_start(m_Frame_Median, Gtk::PACK_START, 5);
	m_VBox_Spatial_Filters.pack_start(m_Frame_Median3D, Gtk::PACK_START, 5);
	m_VBox_Spatial_Filters.pack_start(m_Frame_Gaussian, Gtk::PACK_START, 5);
	m_VBox_Spatial_Filters.pack_start(m_Frame_Bilateral, Gtk::PACK_START, 5);
	m_VBox_Spatial_Filters.pack_start(m_Frame_Morphological, Gtk::PACK_START, 5);
	add(m_VBox_Spatial_Filters);

	set_title("OpenCV Filters");
	set_skip_taskbar_hint(true); // No task bar entry
	set_type_hint(Gdk::WINDOW_TYPE_HINT_DIALOG); // No task bar entry, always on top
	set_size_request(250, 675);
}

guiTools::~guiTools() { }


void guiTools::setSpatialFilter() {
	if(m_medianFiter) {
		frameManager->setFilterMedian(m_kernelRadiusMedian, m_CheckButton_Median_Masked.get_active() );
		controller->getRenderer()->renderFrame();
	}
	else if(m_medianFiter3D) {
		frameManager->setFilterMedian3D(m_CheckButton_Median3D_Masked.get_active() );
		controller->getRenderer()->renderFrame();
	}
	else if(m_gaussianFiter) {
		frameManager->setFilterGaussian(m_kernelRadiusGauss, m_sigmaGauss, cv::BORDER_REPLICATE);
		controller->getRenderer()->renderFrame();
	} 
	else if(m_bilateralFilter) {
		frameManager->setFilterBilateral(m_kernelRadiusBilateral, m_sigmaBilateralColor, m_sigmaBilateralSpace, cv::BORDER_REPLICATE);
		controller->getRenderer()->renderFrame();
	} 
	else if (m_morphologicalFilter) {
		frameManager->setFilterMorphological(m_kernelTypeMorphological, m_kernelRadiusMorphological, m_CheckButton_Morphological_Masked.get_active(), cv::BORDER_REPLICATE);
		controller->getRenderer()->renderFrame();
	}
	else {
		frameManager->setFilterNone();
		controller->getRenderer()->renderFrame();
	}
}


void guiTools::on_checkbutton_median_clicked() {
	m_CheckButton_Median3D.set_active(false);
	m_CheckButton_Bilateral.set_active(false);
	m_CheckButton_Morphological.set_active(false);
	m_medianFiter = !m_medianFiter;
	if(m_medianFiter)	{
		m_CheckButton_Median.set_active(true);
	}
	setSpatialFilter();
}


void guiTools::on_checkbutton_median_masked_clicked() {
	setSpatialFilter();
}

void guiTools::on_spinbutton_median_radius_value_changed() {
	m_kernelRadiusMedian = m_SpinButton_Median_Radius.get_value_as_int();
	setSpatialFilter();
}


void guiTools::on_checkbutton_median3D_clicked() {
	m_CheckButton_Median.set_active(false);
	m_CheckButton_Bilateral.set_active(false);
	m_CheckButton_Morphological.set_active(false);
	m_medianFiter3D = !m_medianFiter3D;
	if(m_medianFiter3D)	{
		m_CheckButton_Median3D.set_active(true);
	}
	setSpatialFilter();
}


void guiTools::on_checkbutton_median3D_masked_clicked() {
	setSpatialFilter();
}


void guiTools::on_checkbutton_gauss_clicked() {
	m_CheckButton_Median.set_active(false);
	m_CheckButton_Median3D.set_active(false);
	m_CheckButton_Bilateral.set_active(false);
	m_CheckButton_Morphological.set_active(false);
	m_gaussianFiter = !m_gaussianFiter;
	if(m_gaussianFiter)	{
		m_CheckButton_Gauss.set_active(true);
	}
	setSpatialFilter();
}


void guiTools::on_spinbutton_gauss_radius_value_changed() {
	m_kernelRadiusGauss = m_SpinButton_Gauss_Radius.get_value_as_int();
	m_HScale_Gauss_Sigma.set_value(frameProcessor->calcGaussianSigma(m_kernelRadiusGauss));
	setSpatialFilter();
}


void guiTools::on_checkbutton_gauss_auto_sigma_clicked() {
	bool manualSigma = m_CheckButton_Gauss_Auto_Sigma.get_active();
	if(manualSigma) {
		m_sigmaGauss = frameProcessor->calcGaussianSigma(m_kernelRadiusGauss);
		m_HScale_Gauss_Sigma.set_value(m_sigmaGauss);
		m_HScale_Gauss_Sigma.set_sensitive(true);
		m_Label_Gauss_Auto_Sigma.set_sensitive(true);
	} else {
		m_sigmaGauss = -1.0;
		m_HScale_Gauss_Sigma.set_value(frameProcessor->calcGaussianSigma(m_kernelRadiusGauss));
		m_HScale_Gauss_Sigma.set_sensitive(false);
		m_Label_Gauss_Auto_Sigma.set_sensitive(false);
	}
	setSpatialFilter();
}


bool guiTools::on_slider_gauss_sigma_change_value(Gtk::ScrollType type, double value) {
	m_sigmaGauss = value;
	setSpatialFilter();
	return true;
}

// Gtk's Hscale widgets normally "jump" to a specific position with a middle-click.
// To achieve this with the left mouse button, the event is manipulated before the widgets reacts to it
bool guiTools::on_slider_gauss_sigma_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}

// See on_slider_gauss_sigma_clicked()
bool guiTools::on_slider_gauss_sigma_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


void guiTools::on_checkbutton_bilateral_clicked() {
	m_CheckButton_Median.set_active(false);
	m_CheckButton_Median3D.set_active(false);
	m_CheckButton_Gauss.set_active(false);
	m_CheckButton_Morphological.set_active(false);
	m_bilateralFilter = !m_bilateralFilter;
	if(m_bilateralFilter)	{
		m_CheckButton_Bilateral.set_active(true);
	}
	setSpatialFilter();
}


void guiTools::on_spinbutton_bilateral_radius_value_changed() {
	m_kernelRadiusBilateral = m_SpinButton_Bilateral_Radius.get_value_as_int();
	setSpatialFilter();
}

bool guiTools::on_slider_bilateral_sigma_color_change_value(Gtk::ScrollType type, double value) {
	m_sigmaBilateralColor = value;
	setSpatialFilter();
	return true;
}


bool guiTools::on_slider_bilateral_sigma_color_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}
bool guiTools::on_slider_bilateral_sigma_color_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


bool guiTools::on_slider_bilateral_sigma_space_change_value(Gtk::ScrollType type, double value) {
	m_sigmaBilateralColor = value;
	setSpatialFilter();
	return true;
}


bool guiTools::on_slider_bilateral_sigma_space_clicked(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


bool guiTools::on_slider_bilateral_sigma_space_released(GdkEventButton* event) {
	if (event->button == 1) { // left click
		event->button = 2; // middle click
	}
	return false;
}


void guiTools::on_checkbutton_morphological_clicked() {
	m_CheckButton_Median.set_active(false);
	m_CheckButton_Gauss.set_active(false);
	m_CheckButton_Bilateral.set_active(false);
	m_morphologicalFilter = !m_morphologicalFilter;
	if(m_morphologicalFilter) {
		m_CheckButton_Morphological.set_active(true);
	}
	setSpatialFilter();
}


void guiTools::on_checkbutton_morphological_masked_clicked() {
	setSpatialFilter();
}


void guiTools::on_spinbutton_morphological_radius_value_changed() {
	m_kernelRadiusMorphological = m_SpinButton_Morphological_Radius.get_value_as_int();
	setSpatialFilter();
}


void guiTools::on_combo_morphological_kerneltype_changed() {
	m_kernelTypeMorphological = m_Combo_Morphological_Kerneltype.get_active_row_number();
	setSpatialFilter();
}
