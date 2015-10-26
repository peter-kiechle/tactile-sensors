#ifndef GUITREEVIEW_H_
#define GUITREEVIEW_H_

#include <gtkmm.h>

class Controller;
class FrameManager;
class FrameProcessor;
class guiMain;

class RowData {

public:
	RowData(Glib::ustring label, bool plot, Glib::ustring value);
	RowData(Glib::ustring label, const std::vector<RowData>& children);
	RowData(const RowData& src);
	~RowData();
	RowData& operator=(const RowData& src);

	Glib::ustring m_label;
	bool m_plot;
	Glib::ustring m_value;
	std::vector<RowData> m_children;
};

/**
 * @class guiTreeView
 * @brief Tree view of matrix characteristics. Follows the MVC pattern.
 */
class guiTreeView : public Gtk::Frame {

public:
	guiTreeView(Controller *c, guiMain *gui);
	virtual ~guiTreeView();


	void init();
	void updateCharacteristics();

private:

	Controller *controller;
	FrameManager *frameManager;
	FrameProcessor *frameProcessor;
	guiMain* mainGUI;

	Gtk::VBox m_VBox;
	Gtk::ScrolledWindow m_ScrolledWindow;
	Gtk::TreeView m_TreeView;
	Glib::RefPtr<Gtk::TreeStore> m_refTreeModel;

	std::vector<RowData> m_rows;

	struct ModelColumns : public Gtk::TreeModelColumnRecord {
		Gtk::TreeModelColumn<Glib::ustring> label;
		Gtk::TreeModelColumn<bool> plot;
		Gtk::TreeModelColumn<Glib::ustring> value;
		Gtk::TreeModelColumn<bool> visible;
		ModelColumns();
	};

	const ModelColumns m_columns;

	void createModel();
	void defineColumns();
	void collectRowData();
	void addData(const RowData& foo);

protected:
	void on_cell_toggled(const Glib::ustring& path_string);
	void notifyMain();
};

#endif /* GUITREEVIEW_H_ */
