#include <boost/lexical_cast.hpp>

#include "controller.h"
#include "guiTreeView.h"
#include "guiMain.h"


RowData::RowData(Glib::ustring label, bool plot, Glib::ustring value) {
	m_label = label;
	m_plot = plot;
	m_value = value;
}


RowData::RowData(Glib::ustring label, const std::vector<RowData>& children) {
	m_label = label;
	m_plot = false;
	m_value = "0";
	m_children = children;
}

RowData::RowData(const RowData& src) {
	operator=(src);
}


RowData::~RowData() { }


RowData& RowData::operator=(const RowData& src) {
	m_label = src.m_label;
	m_plot = src.m_plot;
	m_value = src.m_value;
	m_children = src.m_children;
	return *this;
}


guiTreeView::ModelColumns::ModelColumns() {
	add(label);
	add(plot);
	add(value);
	add(visible);
}


guiTreeView::guiTreeView(Controller *c, guiMain *gui)
: m_VBox(false, 5)
{
	controller = c;
	frameManager = controller->getFrameManager();
	frameProcessor = frameManager->getFrameProcessor();
	mainGUI = gui;

	set_border_width(0);
	set_size_request(275, 400);
	set_shadow_type(Gtk::SHADOW_NONE);

	add(m_VBox);

	m_ScrolledWindow.set_shadow_type(Gtk::SHADOW_ETCHED_IN);
	m_ScrolledWindow.set_policy(Gtk::POLICY_AUTOMATIC, Gtk::POLICY_AUTOMATIC);
	m_VBox.pack_start(m_ScrolledWindow);

	init();

	m_ScrolledWindow.add(m_TreeView);
	show_all();
}

guiTreeView::~guiTreeView() { }


void guiTreeView::init() {

	// Create model
	createModel();

	// Define view
	m_TreeView.set_model(m_refTreeModel);
	m_TreeView.set_rules_hint();
	Glib::RefPtr<Gtk::TreeSelection> refTreeSelection = m_TreeView.get_selection();
	refTreeSelection->set_mode(Gtk::SELECTION_NONE);
	m_TreeView.expand_all();

	defineColumns();

	notifyMain();
}


void guiTreeView::createModel() {

	m_refTreeModel = Gtk::TreeStore::create(m_columns);

	collectRowData();

	std::for_each(
			m_rows.begin(), m_rows.end(),
			sigc::mem_fun(*this, &guiTreeView::addData) );

}


void guiTreeView::collectRowData() {
	m_rows.clear();
	// Add matrix characteristics
	std::vector<RowData> item;
	int frameID = frameManager->getCurrentFrameID();
	for(uint m = 0; m < frameManager->getNumMatrices(); m++) {
		float average = frameProcessor->getMatrixAverage(frameID, m);
		float min = frameProcessor->getMatrixMin(frameID, m);
		float max = frameProcessor->getMatrixMax(frameID, m);
		item.clear();
		item.push_back( RowData("Average Pressure", false, boost::lexical_cast<std::string>(average)) );
		item.push_back( RowData("Min Pressure", false, boost::lexical_cast<std::string>(min)) );
		item.push_back( RowData("Max Pressure", false, boost::lexical_cast<std::string>(max)) );

		std::ostringstream ss;
		ss.clear();
		ss << "Matrix " << m;
		m_rows.push_back( RowData(ss.str(), item) );
	}

	if(frameManager->getTSFrameAvailable()) {
		// Add frame characteristics
		float average = frameProcessor-> getAverage(frameID);
		float min = frameProcessor-> getMin(frameID);
		float max = frameProcessor->getMax(frameID);
		item.clear();
		item.push_back( RowData("Average Pressure", true, boost::lexical_cast<std::string>(average)) );
		item.push_back( RowData("Min Pressure", false, boost::lexical_cast<std::string>(min)) );
		item.push_back( RowData("Max Pressure", false, boost::lexical_cast<std::string>(max)) );
		m_rows.push_back( RowData("All", item) );
	}
}


void guiTreeView::addData(const RowData& parent) {
	// Create new parent row
	Gtk::TreeRow row = *(m_refTreeModel->append());
	row[m_columns.label] = parent.m_label;
	row[m_columns.plot] = parent.m_plot;
	row[m_columns.value] = parent.m_value;
	row[m_columns.visible] = false; // Parent rows have no visible entries

	//Add children
	for(std::vector<RowData>::const_iterator iter = parent.m_children.begin(); iter != parent.m_children.end(); ++iter) {
		const RowData& child = *iter;
		//Create new child row
		Gtk::TreeRow child_row = *(m_refTreeModel->append(row.children()));
		child_row[m_columns.label] = child.m_label;
		child_row[m_columns.plot] = child.m_plot;
		child_row[m_columns.value] = child.m_value;
		child_row[m_columns.visible] = true;
	}
}


void guiTreeView::defineColumns() {

	m_TreeView.remove_all_columns();

	// Characteristics
	{
		int cols_count = m_TreeView.append_column("Characterics",  m_columns.label);
		Gtk::TreeViewColumn* pColumn = m_TreeView.get_column(cols_count-1);
		if(pColumn) {
			Gtk::CellRenderer* pRenderer = pColumn->get_first_cell();
			pRenderer->property_xalign().set_value(0.0); // Left align

			pColumn->set_sizing(Gtk::TREE_VIEW_COLUMN_AUTOSIZE);
			pColumn->set_resizable();
			pColumn->set_clickable();
		}
	}

	// Plot
	{
		int cols_count = m_TreeView.append_column_editable("Plot", m_columns.plot);
		Gtk::TreeViewColumn* pColumn = m_TreeView.get_column(cols_count-1);
		if(pColumn) {
			Gtk::CellRendererToggle* pRenderer = static_cast<Gtk::CellRendererToggle*>(pColumn->get_first_cell());
			pRenderer->property_xalign().set_value(0.0);

			pColumn->add_attribute(pRenderer->property_visible(), m_columns.visible);
			pRenderer->signal_toggled().connect(sigc::mem_fun(*this, &guiTreeView::on_cell_toggled));

			pColumn->set_sizing(Gtk::TREE_VIEW_COLUMN_AUTOSIZE);
			pColumn->set_resizable();
			pColumn->set_clickable();
		}
	}

	// Value
	{
		int cols_count = m_TreeView.append_column_editable("Value", m_columns.value);
		Gtk::TreeViewColumn* pColumn = m_TreeView.get_column(cols_count-1);
		if(pColumn) {
			Gtk::CellRendererToggle* pRenderer = static_cast<Gtk::CellRendererToggle*>(pColumn->get_first_cell());
			pRenderer->property_xalign().set_value(1.0); // Right align
			pColumn->add_attribute(pRenderer->property_visible(), m_columns.visible);

			pColumn->set_sizing(Gtk::TREE_VIEW_COLUMN_AUTOSIZE);
			pColumn->set_resizable();
			pColumn->set_clickable();
		}
	}
}


void guiTreeView::on_cell_toggled(const Glib::ustring& path_string) {

	// Get the model row that has been toggled.
	Gtk::TreeModel::Path path = Gtk::TreeModel::Path(path_string);
	Gtk::TreeModel::Row row = *m_refTreeModel->get_iter(path);
	bool value = row[m_columns.plot];
	int parentID = path[0];
	int childID = path[1];
	printf("%s parentID: %d, childID: %d, Value: %d\n", path_string.c_str(), parentID, childID, value);

	notifyMain();
	mainGUI->updateDataset();
}


void guiTreeView::notifyMain() {
	// Build  collection of selected characteristics
	std::vector<std::vector<int> > characteristics;

	uint indexOuter = 0; // Index of iterator
	Gtk::TreeModel::Children nodes = m_refTreeModel->children();
	for(Gtk::TreeModel::Children::iterator iterOuter = nodes.begin(); iterOuter != nodes.end(); ++iterOuter, ++indexOuter) {
		Gtk::TreeModel::Row nodeRow = *iterOuter;
		Gtk::TreeModel::Children children = nodeRow.children();
		int indexInner = 0; // Index of iterator
		for(Gtk::TreeModel::Children::iterator iterInner = children.begin(); iterInner != children.end(); ++iterInner, ++indexInner) {
			Gtk::TreeModel::Row childRow = *iterInner;
			// cout << childRow[m_columns.label] << " " << childRow[m_columns.plot] << " " << childRow[m_columns.value] << endl;
			if(childRow[m_columns.plot] == true) {
				std::vector<int> id(2);
				id[0] = indexOuter;
				id[1] = indexInner;
				characteristics.push_back(id);
			}
		}
	}
	mainGUI->setCharacteristics(characteristics);
}


void guiTreeView::updateCharacteristics() {
	// Update the model
	int frameID = frameManager->getCurrentFrameID();
	uint indexOuter = 0; // Index of iterator
	Gtk::TreeModel::Children nodes = m_refTreeModel->children();
	for(Gtk::TreeModel::Children::iterator iterOuter = nodes.begin(); iterOuter != nodes.end(); ++iterOuter, ++indexOuter) {
		Gtk::TreeModel::Row nodeRow = *iterOuter;
		Gtk::TreeModel::Children children = nodeRow.children();
		int indexInner = 0; // Index of iterator
		float value;
		for(Gtk::TreeModel::Children::iterator iterInner = children.begin(); iterInner != children.end(); ++iterInner, ++indexInner) {
			Gtk::TreeModel::Row childRow = *iterInner;
			if(indexOuter < frameManager->getNumMatrices()) { // Matrix Characteristics
				switch(indexInner) {
				case 0:
					value = frameProcessor->getMatrixAverage(frameID, indexOuter); break;
				case 1:
					value = frameProcessor->getMatrixMin(frameID, indexOuter); break;
				case 2:
					value = frameProcessor->getMatrixMax(frameID, indexOuter); break;
				default:
					value = 0; break;
				}
			} else { // Frame Characteristics
				switch(indexInner)	{
				case 0:
					value = frameProcessor->getAverage(frameID); break;
				case 1:
					value = frameProcessor->getMin(frameID); break;
				case 2:
					value = frameProcessor->getMax(frameID); break;
				default:
					value = 0; break;
				}
			}
			ostringstream ss;
			ss << std::fixed << std::setprecision(2) << value;
			childRow[m_columns.value] = ss.str();
		}
	}
}
