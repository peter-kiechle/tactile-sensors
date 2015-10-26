/**
 * @file
 * Taken from SDHLibrary (modified version).
 */

//======================================================================
/*!
    \file
     \section sdhlibrary_cpp_demo_sdhoptions_h_general General file information

       \author   Dirk Osswald
       \date     2008-05-05

     \brief
       Implementation of a class to parse common %SDH related command line options

     \section sdhlibrary_cpp_demo_sdhoptions_h_copyright Copyright

     Copyright (c) 2008 SCHUNK GmbH & Co. KG

     <HR>
     \internal

       \subsection sdhlibrary_cpp_demo_sdhoptions_h_details SVN related, detailed file specific information:
         $LastChangedBy: Osswald2 $
         $LastChangedDate: 2011-02-07 09:15:17 +0100 (Mo, 07 Feb 2011) $
         \par SVN file revision:
           $Id: sdhoptions.h 6424 2011-02-07 08:15:17Z Osswald2 $

     \subsection sdhlibrary_cpp_demo_sdhoptions_h_changelog Changelog of this file:
         \include sdhoptions.h.log

*/
//======================================================================

#ifndef SDHOPTIONS_h
#define SDHOPTIONS_h

//----------------------------------------------------------------------
// System Includes - include with <>
//----------------------------------------------------------------------

#include <getopt.h>
#include <assert.h>

#include <iostream>
#include <string>

//----------------------------------------------------------------------
// Project Includes - include with ""
//---------------------------------------------------------------------

#include <sdh/sdh.h>

//----------------------------------------------------------------------
// Defines, enums, unions, structs,
//----------------------------------------------------------------------

/*! string defining all the usage helptexts included by default
 *
 *  \bug When compiled with VCC then the macros WITH_ESD_CAN / WITH_PEAK_CAN
 *  used above are not available since these are defined in the VCC project
 *  settings of the SDHLibrary VCC-Project. Therefore the value of SDHUSAGE_DEFAULT
 *  is incorrect and thus the cSDHOptions will display an incomplete
 *  usage string when called with -h/--help.<br>
 *  Workaround: use the online help contained in the doxygen documentation:
 *  <a href="./group__sdh__library__cpp__onlinehelp__group.html">Online help of demonstration programs</a>
 */
#define SDHUSAGE_DEFAULT "general sdhcom_serial sdhcom_common sdhcom_esdcan sdhcom_peakcan sdhcom_cancommon sdhcom_tcp"

#define SDH_DEFAULT_TCP_ADR  "192.168.1.1"
#define SDH_DEFAULT_TCP_PORT 23

//----------------------------------------------------------------------
// Global variables
//----------------------------------------------------------------------


/*!
 * \brief class for command line option parsing holding option parsing results
 */
class cSDHOptions
{
public:
    static int const MAX_DEV_LENGTH = 32;

    std::string   usage;

    int           debug_level;
    std::ostream* debuglog;

    int           sdhport;
    char          sdh_rs_device[MAX_DEV_LENGTH];
    double        timeout;
    unsigned long rs232_baudrate;

    bool          use_can_esd;
    int           net;

    bool          use_can_peak;
    char          sdh_canpeak_device[MAX_DEV_LENGTH];

    unsigned long can_baudrate;
    unsigned int  id_read;
    unsigned int  id_write;

    bool          use_radians;
    bool          use_fahrenheit;
    double        period;

    int           dsaport;
    char          dsa_rs_device[MAX_DEV_LENGTH];
    bool          do_RLE;

    int           framerate;
    bool          fullframe;
    bool          sensorinfo;
    bool          controllerinfo;
    int           matrixinfo[6];

    double        sensitivity[6];
    unsigned int  threshold[6];
    bool          reset_to_default;
    bool          persistent;
    bool          showdsasettings;

    bool          use_tcp;
    std::string   tcp_adr;
    int           tcp_port;

    /*!
     * constructor: init members to their default values
     *
     * \param option_selection - string that names the options to
     *        include in helptext for online help. With a text including one of
     *        the following keywords the corresponding helptext is added
     *        to the usage helptext
     *        - "general" see sdhusage_general
     *        - "sdhcom_serial" see sdhusage_sdhcom_serial
     *        - "sdhcom_common" see sdhusage_sdhcom_common
     *        - "sdhcom_esdcan" see sdhusage_sdhcom_esdcan
     *        - "sdhcom_peakcan" see sdhusage_sdhcom_peakcan
     *        - "sdhcom_cancommon" see sdhusage_sdhcom_cancommon
     *        - "sdhcom_tcp" see sdhusage_sdhcom_tcp
     *        - "sdhother" see sdhusage_sdhother
     *        - "dsacom" see sdhusage_dsacom
     *        - "dsaother" see sdhusage_dsaother
     */
    cSDHOptions( char const* option_selection = SDHUSAGE_DEFAULT );

    //! destructor, clean up
    ~cSDHOptions();

    /*! parse the command line parameters \a argc, \a argv into members. \a helptext, \a progname, \a version, \a libname and \a librelease are used when printing online help.
        start parsing at option with index *p_option_index
        parse all options if parse_all is true, else only one option is parsed

        \return the optind index of the first non option argument in argv
     */
    int Parse( int argc, char** argv,
               char const* helptext, char const* progname, char const* version, char const* libname, char const* librelease );

    /*!
     * convenience function to open the communication of the
     * given \a hand object according to the parsed parameters.
     *
     * @param hand - reference to a cSDH object to open
     */
    void OpenCommunication( NS_SDH cSDH &hand );
};

#endif

//======================================================================
/*
  Here are some settings for the emacs/xemacs editor (and can be safely ignored):
  (e.g. to explicitely set C++ mode for *.h header files)

  Local Variables:
  mode:C++
  mode:ELSE
  End:
*/
//======================================================================
