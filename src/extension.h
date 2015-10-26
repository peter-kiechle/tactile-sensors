#ifndef EXTENSION_H_
#define EXTENSION_H_

#include <iostream>
#include <sstream>
#include <list>

#include "sdh/dbg.h"
#include "sdh/rs232-cygwin.h"
#include "sdh/basisdef.h"
#include "sdh/crc.h"

#include "sdh/dsa.h"
#include "sdh/dsa.h"

// To list comports
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/serial.h>

using namespace std;
USING_NAMESPACE_SDH


/**
 * @class ExtException
 * @brief Derived exception class for low-level DSA related exceptions.
 */
class ExtException: public cSDHLibraryException
{
public:
	ExtException( cMsg const & _msg )
	: cSDHLibraryException( "Library Extension Exception", _msg )
	{}
};

enum DeviceType {SDH2=0, DSACON32m=1};

/**
 * @class Device
 * @brief Simple structure to manage the device, its type and serial number.
 */
class Device
{
public:

	Device(string name, DeviceType type, UInt32 serial) { device_name = name, device_type = type; serial_no = serial; device_format_string = ""; }
	Device(string name, DeviceType type, UInt32 serial, string format_string) { device_name = name, device_type = type; serial_no = serial; device_format_string = format_string; }
	~Device() {};

	string device_name;
	DeviceType device_type;
	UInt32 serial_no;
	string device_format_string;
};

/**
 * @class Ext
 * @brief Extension to the SDH-Library to automatically query all available comports for connected DSA / SDH devices
 * @note Code is partially taken from the SDH-Library
 */
class Ext
{
public:

	/// Error codes returned by the remote DSACON32m tactile sensor controller
	enum eDSAErrorCode
	{
		E_SUCCESS,
		E_NOT_AVAILABLE,
		E_NO_SENSOR,
		E_NOT_INITIALIZED,
		E_ALREADY_RUNNING,
		E_FEATURE_NOT_SUPPORTED,
		E_INCONSISTENT_DATA,
		E_TIMEOUT,
		E_READ_ERROR,
		E_WRITE_ERROR,
		E_INSUFFICIENT_RESOURCES,
		E_CHECKSUM_ERROR,
		E_CMD_NOT_ENOUGH_PARAMS,
		E_CMD_UNKNOWN,
		E_CMD_FORMAT_ERROR,
		E_ACCESS_DENIED,
		E_ALREADY_OPEN,
		E_CMD_FAILED,
		E_CMD_ABORTED,
		E_INVALID_HANDLE,
		E_DEVICE_NOT_FOUND,
		E_DEVICE_NOT_OPENED,
		E_IO_ERROR,
		E_INVALID_PARAMETER,
		E_INDEX_OUT_OF_BOUNDS,
		E_CMD_PENDING,
		E_OVERRUN,
		E_RANGE_ERROR
	};

	/// A data structure describing the controller info about the remote DSACON32m controller
	struct sControllerInfo
	{
		UInt16 error_code;
		UInt32 serial_no;
		UInt8  hw_version;
		UInt16 sw_version;
		UInt8  status_flags;
		UInt8  feature_flags;
		UInt8  senscon_type;
		UInt8  active_interface;
		UInt32 can_baudrate;
		UInt16 can_id;
	}   SDH__attribute__((__packed__)); // for gcc we have to set the necessary 1 byte packing with this attribute

private:

	/// Data structure for storing responses from the remote DSACON32m controller
	struct sResponse {
		UInt8   packet_id;
		UInt16  size;
		UInt8*  payload;
		Int32   max_payload_size;

		/// constructor to init pointer and max size
		sResponse( UInt8* _payload, int _max_payload_size )
		{
			payload = _payload;
			max_payload_size = _max_payload_size;
		}
	} SDH__attribute__((__packed__));  // for gcc we have to set the necessary 1 byte packing with this attribute

	friend std::ostream &operator<<(std::ostream &stream, Ext::sResponse const &response );

    /// A stream object to print coloured debug messages
    cDBG dbg;

	/// flag, true if data should be sent Run Length Encoded by the remote DSACON32m controller
	bool do_RLE;

	/// List of expected devices
	std::list<Device*>& deviceList;

	void WriteCommandDSA(UInt8 command, UInt8* payload, UInt16 payload_len, cRS232& comm_interface)	throw (ExtException*);

	inline void WriteCommandDSA(UInt8 command, cRS232& comm_interface){
		WriteCommandDSA(command, NULL, 0, comm_interface);
	}

	void WriteCommandSDH(string command, cRS232& comm_interface) throw (ExtException*);

	int ReadBytes(UInt8 *data, cRS232& comm_interface, int max_size) throw (ExtException*);

	UInt32 ExtractSerialDSA(UInt8 *data, int len);
	UInt32 ExtractSerialSDH(UInt8 *data, int len);

	void FlushInput( long timeout_us_first, long timeout_us_subsequent, cRS232& comm_interface);


public:

	/// Constructor
	Ext(int debug_level, std::list<Device*>& _deviceList);

	/// Destructor: clean up and delete dynamically allocated memory
	~Ext();

	/**
	 * (Re-)open connection to DSACON32m controller, this is called by the constructor automatically, but is still useful to call after a call to Close()
	 * @return void
	 */
	void Open()	throw (ExtException*);

	/**
	 * Set the framerate of the remote DSACON32m controller to 0 and close connection to it.
     * @return void
	 */
	void Close() throw (ExtException*);

	/**
	 * Enumeration of available com ports
	 * Credits go to Søren Holm: http://stackoverflow.com/questions/2530096/how-to-find-all-serial-devices-ttys-ttyusb-on-linux-without-opening-them
	 * @param tty The tty-path
	 * @return The driver name
	 */
	string getComportDriver(const string& tty);

	/**
	 * Register the available device
	 * Credits go to Søren Holm: http://stackoverflow.com/questions/2530096/how-to-find-all-serial-devices-ttys-ttyusb-on-linux-without-opening-them
	 * @param comList The final list of devices.
	 * @param comList8250 A separate list of serial8250-devices
	 * @param dir The device directory
	 * @return void
	 */
	void addComport(list<string>& comList, list<string>& comList8250, const string& dir);

	/**
	 * Serial8250-devices must be probe to check for validity
	 * Credits go to Søren Holm: http://stackoverflow.com/questions/2530096/how-to-find-all-serial-devices-ttys-ttyusb-on-linux-without-opening-them
	 * @param comList The final list of devices.
	 * @param comList8250 A separate list of serial8250-devices
	 * @return void
	 */
	void probeSerial8250Comports(list<string>& comList, list<string> comList8250);

	/**
	 * List available comports
	 * Credits go to Søren Holm: http://stackoverflow.com/questions/2530096/how-to-find-all-serial-devices-ttys-ttyusb-on-linux-without-opening-them
	 * @return The device list
	 */
	list<string> listComports();

	/**
	 * Auto-detect available tty devices based on response and serial number
	 * @return void
	 */
	void IdentifyDevices();

};

#endif /* EXTENSION_H_ */
