
#ifndef GUITOOLS_H_
#define GUITOOLS_H_

#include <gtkmm.h>

#include "controller.h"

class Controller;
class guiMain;

/**
 * @class guiTools
 * @brief Dialog for OpenCV filters.
 */
class guiTools : public Gtk::Window {
public:
	guiTools(Controller *c, guiMain *gui);
	virtual ~guiTools();

private:

	Controller *controller;
	FrameManager *frameManager;
	FrameProcessor *frameProcessor;
	guiMain *mainGUI;

	Gtk::VBox m_VBox_Spatial_Filters;

	bool m_medianFiter;
	bool m_medianFiter3D;
	bool m_gaussianFiter;
	bool m_bilateralFilter;
	bool m_morphologicalFilter;


	// 2D Median filter
	Gtk::Frame m_Frame_Median;
	Gtk::VBox m_VBox_Median;
	Gtk::CheckButton m_CheckButton_Median;
	Gtk::CheckButton m_CheckButton_Median_Masked;

	Gtk::HBox m_HBox_Median_Radius;
	Gtk::Label m_Label_Median_Radius;
	Gtk::Adjustment m_Adjustment_Median_Radius;
	Gtk::SpinButton m_SpinButton_Median_Radius;
	int m_kernelRadiusMedian;


	// 3D Median filter
	Gtk::Frame m_Frame_Median3D;
	Gtk::VBox m_VBox_Median3D;
	Gtk::CheckButton m_CheckButton_Median3D;
	Gtk::CheckButton m_CheckButton_Median3D_Masked;


	// Gaussian filter
	Gtk::Frame m_Frame_Gaussian;
	Gtk::VBox m_VBox_Gauss;
	Gtk::CheckButton m_CheckButton_Gauss;

	Gtk::HBox m_HBox_Gauss_Radius;
	Gtk::Label m_Label_Gauss_Radius;
	Gtk::Adjustment m_Adjustment_Gauss_Radius;
	Gtk::SpinButton m_SpinButton_Gauss_Radius;

	Gtk::CheckButton m_CheckButton_Gauss_Auto_Sigma;
	Gtk::HBox m_HBox_Gauss_Auto_Sigma;
	Gtk::Label m_Label_Gauss_Auto_Sigma;
	Gtk::Adjustment m_Adjustment_Gauss_Sigma;
	Gtk::HScale m_HScale_Gauss_Sigma;

	int m_kernelRadiusGauss;
	double m_sigmaGauss;

	// Bilateral filter
	Gtk::Frame m_Frame_Bilateral;
	Gtk::VBox m_VBox_Bilateral;
	Gtk::CheckButton m_CheckButton_Bilateral;

	Gtk::HBox m_HBox_Bilateral_Radius;
	Gtk::Label m_Label_Bilateral_Radius;
	Gtk::Adjustment m_Adjustment_Bilateral_Radius;
	Gtk::SpinButton m_SpinButton_Bilateral_Radius;

	Gtk::HBox m_HBox_Bilateral_Sigma_Color;
	Gtk::Label m_Label_Bilateral_Sigma_Color;
	Gtk::Adjustment m_Adjustment_Bilateral_Sigma_Color;
	Gtk::HScale m_HScale_Bilateral_Sigma_Color;

	Gtk::HBox m_HBox_Bilateral_Sigma_Space;
	Gtk::Label m_Label_Bilateral_Sigma_Space;
	Gtk::Adjustment m_Adjustment_Bilateral_Sigma_Space;
	Gtk::HScale m_HScale_Bilateral_Sigma_Space;

	int m_kernelRadiusBilateral;
	double m_sigmaBilateralColor;
	double m_sigmaBilateralSpace;


	// Opening morphological transformation
	Gtk::Frame m_Frame_Morphological;
	Gtk::VBox m_VBox_Morphological;
	Gtk::CheckButton m_CheckButton_Morphological;
	Gtk::CheckButton m_CheckButton_Morphological_Masked;

	Gtk::HBox m_HBox_Morphological_Radius;
	Gtk::Label m_Label_Morphological_Radius;
	Gtk::Adjustment m_Adjustment_Morphological_Radius;
	Gtk::SpinButton m_SpinButton_Morphological_Radius;

	Gtk::HBox m_HBox_Morphological_Kerneltype;
	Gtk::Label m_Label_Morphological_Kerneltype;
	Gtk::ComboBoxText m_Combo_Morphological_Kerneltype;

	int m_kernelRadiusMorphological;
	int m_kernelTypeMorphological;


	void setSpatialFilter();

	void on_checkbutton_median_clicked();
	void on_checkbutton_median_masked_clicked();
	void on_spinbutton_median_radius_value_changed();

	void on_checkbutton_median3D_clicked();
	void on_checkbutton_median3D_masked_clicked();

	void on_checkbutton_gauss_clicked();
	void on_spinbutton_gauss_radius_value_changed();
	void on_checkbutton_gauss_auto_sigma_clicked();
	bool on_slider_gauss_sigma_clicked(GdkEventButton* event);
	bool on_slider_gauss_sigma_released(GdkEventButton* event);
	bool on_slider_gauss_sigma_change_value(Gtk::ScrollType type, double value);

	void on_checkbutton_bilateral_clicked();
	void on_spinbutton_bilateral_radius_value_changed();
	bool on_slider_bilateral_sigma_color_clicked(GdkEventButton* event);
	bool on_slider_bilateral_sigma_color_released(GdkEventButton* event);
	bool on_slider_bilateral_sigma_color_change_value(Gtk::ScrollType type, double value);
	bool on_slider_bilateral_sigma_space_clicked(GdkEventButton* event);
	bool on_slider_bilateral_sigma_space_released(GdkEventButton* event);
	bool on_slider_bilateral_sigma_space_change_value(Gtk::ScrollType type, double value);

	void on_checkbutton_morphological_clicked();
	void on_checkbutton_morphological_masked_clicked();
	void on_spinbutton_morphological_radius_value_changed();
	void on_combo_morphological_kerneltype_changed();
};

#endif /* GUITOOLS_H_ */
