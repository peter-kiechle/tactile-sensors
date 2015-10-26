//======================================================================
/*!
    \file

       \author   Dirk Osswald
       \date     2008-05-05

     \brief
       Implementation of a class to parse common %SDH related command line options

     \section sdhlibrary_cpp_demo_sdhoptions_cpp_copyright Copyright

     Copyright (c) 2008 SCHUNK GmbH & Co. KG

     <HR>
     \internal

       \subsection sdhlibrary_cpp_demo_sdhoptions_cpp_details SVN related, detailed file specific information:
         $LastChangedBy: Osswald2 $
         $LastChangedDate: 2011-03-01 17:53:56 +0100 (Di, 01 Mrz 2011) $
         \par SVN file revision:
           $Id: sdhoptions.cpp 6500 2011-03-01 16:53:56Z Osswald2 $

     \subsection sdhlibrary_cpp_demo_sdhoptions_cpp_changelog Changelog of this file:
         \include sdhoptions.cpp.log

*/
//======================================================================

//----------------------------------------------------------------------
// System Includes - include with <>
//----------------------------------------------------------------------

#include <getopt.h>
#include <assert.h>

#include <iostream>
#include <fstream>

using namespace std;

//----------------------------------------------------------------------
// Project Includes - include with ""
//---------------------------------------------------------------------

#include "sdh/sdh.h"
#include "sdh/sdhlibrary_settings.h"
#include "sdh/release.h"
#include "sdh/dsa.h"
#include "sdhoptions.h"

//----------------------------------------------------------------------
// Defines, enums, unions, structs,
//----------------------------------------------------------------------

USING_NAMESPACE_SDH

/*!
 * macro for stringification of \a _x
 *
 * allows to stringify the \b value of a macro:
 * \code
 *   #define foo 4
 *
 *   STRINGIFY( foo ) // yields "foo"
 *   XSTRINGIFY( foo ) // yields "4"
 *  \endcode
 */
#define XSTRINGIFY(_x) STRINGIFY(_x)

//! helper macro for XSTRINGIFY, see there
#define STRINGIFY(_s) #_s

//----------------------------------------------------------------------
// Global variables
//----------------------------------------------------------------------


//! general options
static char const* sdhusage_general =
    //2345678901234567890123456789012345678901234567890123456789012345678901234567890
    //        1         2         3         4         5         6         7         8
    "\nGeneral options:\n"
    "  -h, --help\n"
    "      Show this help message and exit.\n"
    "      \n"
    "  -v, --version\n"
    "      Print the version (revision/release names) and dates of application,\n"
    "      library (and the attached SDH firmware, if found), then exit.\n"
    "      \n"
    "  -V, --version_check\n"
    "      Check the firmware release of the connected SDH if it is the one\n"
    "      recommended by this library. A message will be printed accordingly.\n"
    "      \n"
    "  -d LEVEL, --debug=LEVEL\n"
    "      Print debug messages of level LEVEL or lower while executing the program.\n"
    "      Level 0 (default): No messages,  1: application-level messages, \n"
    "      2: cSDH-level messages, 3: cSDHSerial-level messages\n"
    "      \n"
    "  -l LOGFILE, --debuglog=LOGFILE\n"
    "      Redirect the printed debug messages to LOGFILE instead of default \n"
    "      standard error. If LOGFILE starts with '+' then the output will be \n"
    "      appended to the file (without the leading '+'), else the file will be\n"
    "      overwritten.\n"
    "      \n"
    ;

//! RS232 communication options
static char const* sdhusage_sdhcom_serial =
    "\nCommunication options:\n"
    "  -p PORT, --port=PORT, --sdhport=PORT\n"
    "      Use RS232 communication PORT to connect to the SDH instead of the default\n"
    "      0='COM1'='/dev/ttyS0'.\n"
    "      \n"
#if ! SDH_USE_VCC
    "  --sdh_rs_device=DEVICE_FORMAT_STRING\n"
    "      Use DEVICE_FORMAT_STRING instead of the default \"/dev/ttyS%d\". Useful\n"
    "      e.g. to use USB to RS232 converters available via \"/dev/ttyUSB%d\". \n"
    "      If the DEVICE_FORMAT_STRING contains '%d' then the PORT must also be \n"
    "      provided. If not then the DEVICE_FORMAT_STRING is the full device name. \n"
    "      \n"
#endif
    ;

//! Common communication options
static char const* sdhusage_sdhcom_common =
    "  -T TIMEOUT, --timeout=TIMEOUT Timeout in seconds when waiting for data from\n"
    "      SDH. The default -1 means: wait forever.\n"
    "      \n"
    "  -b BAUDRATE, --baud=BAUDRATE\n"
    "      Use BAUDRATE in bit/s for communication. Default=115200 Bit/s for RS232\n"
#if WITH_ESD_CAN || WITH_PEAK_CAN
    "      and 1000000 Bit/s (1MBit/s) for CAN\n"
#endif
    ;

//! ESD CAN communication options
static char const* sdhusage_sdhcom_esdcan =
#if WITH_ESD_CAN
    "  -c, --can, --canesd\n"
    "      Use CAN bus via an ESD adapter to connect to the SDH instead of RS232.\n"
    "      \n"
    "  -n NET, --net=NET\n"
    "      Use ESD CAN NET for CAN communication, default=0.\n"
    "      \n"
#else
    ""
#endif
    ;
//! PEAK CAN communication options
static char const* sdhusage_sdhcom_peakcan =
#if WITH_PEAK_CAN
    "  --canpeak\n"
    "      Use CAN bus via a PEAK adapter to connect to the SDH instead of RS232.\n"
    "      \n"
#else
    ""
#endif
#if WITH_PEAK_CAN &&  defined( OSNAME_LINUX )
    "  --sdh_canpeak_device=DEVICE_NAME\n"
    "      Use DEVICE_NAME instead of the default \"/dev/pcanusb0\"."
    "      \n"
#endif
    ;
//! Common CAN communication options
static char const* sdhusage_sdhcom_cancommon =
#if WITH_ESD_CAN || WITH_PEAK_CAN
    "  -e ID_READ, --id_read=ID_READ\n"
    "      Use CAN ID ID_READ for receiving CAN messages (default: 43=0x2B).\n"
    "      \n"
    "  -w ID_WRITE, --id_write=ID_WRITE\n"
    "      Use CAN ID ID_WRITE for writing CAN messages (default: 42=0x2A).\n"
    "      \n"
#else
    ""
#endif
    ;

//! TCP communication options
static char const* sdhusage_sdhcom_tcp =
    "  --tcp=IP_OR_HOSTNAME[:PORT]\n"
    "      use TCP for communication with the SDH. The SDH can be reached via\n"
    "      TCP/IP on port PORT at IP_OR_HOSTNAME, which can be a numeric IPv4\n"
    "      address or a hostname. The default is \"" SDH_DEFAULT_TCP_ADR  ":" XSTRINGIFY(SDH_DEFAULT_TCP_PORT) "\"\n"
    "      \n"
    ;


//! Other options
static char const* sdhusage_sdhother =
    "\nOther options:\n"
    "  -R, --radians\n"
    "      Use radians and radians per second for angles and angular velocities\n"
    "      instead of default degrees and degrees per second.\n"
    "      \n"
    "  -F, --fahrenheit\n"
    "      Use degrees fahrenheit to report temperatures instead of default degrees\n"
    "      celsius.\n"
    "      \n"
    "  -t PERIOD, --period=PERIOD\n"
    "      For periodic commands only: Time period of measurements in seconds. The\n"
    "      default of '0' means: report once only. If set then the time since start\n"
    "      of measurement is printed at the beginning of every line.\n"
    "      \n"
    ;
//! DSA (tactile sensor) communication options
static char const* sdhusage_dsacom =
    "\nDSA options (tactile sensor):\n"
    "  -q PORT, --dsaport=PORT\n"
    "      use RS232 communication PORT to connect to to  tactile sensor controller\n"
    "      of SDH  instead of default 1='COM2'='/dev/ttyS1'.\n"
    "      \n"
#if ! SDH_USE_VCC
    "  --dsa_rs_device=DEVICE_FORMAT_STRING\n"
    "      Use DEVICE_FORMAT_STRING instead of the default \"/dev/ttyS%d\". Useful\n"
    "      e.g. to use USB to RS232 converters available via \"/dev/ttyUSB%d\".  If \n"
    "      the DEVICE_FORMAT_STRING contains '%d' then the dsa PORT must also be \n"
    "      provided. If not then the DEVICE_FORMAT_STRING is the full device name.\n"
    "      \n"
#endif
    "  --no_rle\n"
    "      Do not use the RunLengthEncoding\n"
    "      \n"
    ;
//! DSA (tactile sensor) other options
static char const* sdhusage_dsaother =
    "  -r, --framerate=FRAMERATE\n"
    "      Framerate for acquiring full tactile sensor frames.  Default value 0\n"
    "      means 'acquire a single frame only'.  Any value > 0 will make the\n"
    "      DSACON32m controller in the SDH send data at the highest possible rate \n"
    "      (ca. 30 FPS (frames per second)).\n"
    "      \n"
    "  -f, --fullframe\n"
    "      Print acquired full frames numerically.\n"
    "      \n"
    "  -S, --sensorinfo\n"
    "      Print sensor info from DSA (texel dimensions, number of texels...).\n"
    "      \n"
    "  -C, --controllerinfo\n"
    "      Print controller info from DSA (version...).\n"
    "      \n"
    "  -M, --matrixinfo=MATRIX_INDEX\n"
    "      Print matrix info for matrix with index MATRIX_INDEX from DSA.\n"
    "      This includes the current setting for sensitivity and threshold\n"
    "      of the addressed matrix (if supported by the tactile sensor firmware)."
    "      This option can be used multiple times.\n"
    ;

//! DSA (tactile sensor) adjustment options
static char const* sdhusage_dsaadjust =
    "  --sensitivity=SENSITIVITY\n"
    "      Set the sensor sensitivity for all tactile sensor pads to the given\n"
    "      value. The value SENSITIVITY must be in range [0.0 .. 1.0], where\n"
    "      0.0 is minimum and 1.0 is maximum sensitivity.\n"
    "      If --reset is given as well then SENSITIVITY is ignored and\n"
    "      the sensitivity is reset to the factory default.\n"
    "      To see the current setting for sensitivity use -M --matrixinfo\n"
    "      For setting sensitivities individually for a specific sensor X [0..5]\n"
    "      use --sensitivityX=SENSITIVITY\n"
    "\n"
    "  --sensitivityX=SENSITIVITY\n"
    "      X is a sensor index in range [0..5]. Set sensor sensitivity for a\n"
    "      a specific sensor X. See also --sensitivity. \n"
    "      This option can be used multiple times (with different X).\n"
    "      \n"
    "  --threshold=THRESHOLD\n"
    "      Set the sensor threshold for all tactile sensor pads to the given \n"
    "      value. The value THRESHOLD must be in range [0 .. 4095], where\n"
    "      (0 is minimum, 4095 is maximum threshold).\n"
    "      If --reset is given as well then THRESHOLD is ignored and\n"
    "      the threshold is reset to the factory default."
    "      \n"
    "  --thresholdX=THRESHOLD\n"
    "      Set sensor threshold for a specific sensor X.\n"
    "      X is a sensor index in range [0..5]. See also option --threshold.\n"
    "      This option can be used multiple times (with different X).\n"
    "      \n"
    "  --reset\n"
    "      If given, then the values given with --sensitivity(X) \n"
    "      and/or --threshold(X) are reset to their factory default.\n"
    "      \n"
    "  --persistent\n"
    "      If given, then all the currently set values for sensitivity\n"
    "      and threshold are saved persistently in the configuration\n"
    "      memory of the DSACON32m controller in the SDH.\n"
    "      PLEASE NOTE: the maximum write endurance of the configuration memory\n"
    "      is about 100.000 times!\n"
    "      \n"
    "  --showdsasettings\n"
    "      If given, then current settings for sensitivity and\n"
    "      threshold will be printed on stdout first.\n"
    "      \n"
    ;

//! short command line options accepted by the cSDHOptions class
static char const* sdhoptions_short_options = "hvVd:l:p:T:b:cn:e:w:RFt:q:r:fSCM:";
//! long command line options accepted by the cSDHOptions class
static struct option sdhoptions_long_options[] =
{
  // name              , has_arg, flag, val
  {"help"              , 0      , 0   , 'h'      },
  {"version"           , 0      , 0   , 'v'      },
  {"version_check"     , 0      , 0   , 'V'      },
  {"debug"             , 1      , 0   , 'd'      },
  {"debuglog"          , 1      , 0   , 'l'      },

  {"port"              , 1      , 0   , 'p'      },
  {"sdhport"           , 1      , 0   , 'p'      },
  {"sdh_rs_device"     , 1      , 0   , 'S' + 256},
  {"timeout"           , 1      , 0   , 'T'      },
  {"baud"              , 1      , 0   , 'b'      },

  {"can"               , 0      , 0   , 'c'      },
  {"canesd"            , 0      , 0   , 'c'      },
  {"net"               , 1      , 0   , 'n'      },

  {"canpeak"           , 0      , 0   , 'p' + 256},
  {"sdh_canpeak_device", 1      , 0   , 'P' + 256},

  {"id_read"           , 1      , 0   , 'e'      },
  {"id_write"          , 1      , 0   , 'w'      },

  {"tcp"               , 1      , 0   , 'T' + 256 },

  {"radians"           , 0      , 0   , 'R'      },
  {"fahrenheit"        , 0      , 0   , 'F'      },
  {"period"            , 1      , 0   , 't'      },

  {"dsaport"           , 1      , 0   , 'q'      },
  {"dsa_rs_device"     , 1      , 0   , 'D' + 256},
  {"no_rle"            , 0      , 0   , 'r' + 256},

  {"framerate"         , 1      , 0   , 'r'      },
  {"fullframe"         , 0      , 0   , 'f'      },
  {"sensorinfo"        , 0      , 0   , 'S'      },
  {"controllerinfo"    , 0      , 0   , 'C'      },
  {"matrixinfo"        , 1      , 0   , 'M'      },

  {"sensitivity"       , 1      , 0   , 999      },
  {"sensitivity0"      , 1      , 0   , 1000     },
  {"sensitivity1"      , 1      , 0   , 1001     },
  {"sensitivity2"      , 1      , 0   , 1002     },
  {"sensitivity3"      , 1      , 0   , 1003     },
  {"sensitivity4"      , 1      , 0   , 1004     },
  {"sensitivity5"      , 1      , 0   , 1005     },
  {"threshold"         , 1      , 0   , 1009     },
  {"threshold0"        , 1      , 0   , 1010     },
  {"threshold1"        , 1      , 0   , 1011     },
  {"threshold2"        , 1      , 0   , 1011     },
  {"threshold3"        , 1      , 0   , 1013     },
  {"threshold4"        , 1      , 0   , 1014     },
  {"threshold5"        , 1      , 0   , 1015     },
  {"reset"             , 0      , 0   , 'R' + 512},
  {"persistent"        , 0      , 0   , 'P' + 512},
  {"showdsasettings"   , 0      , 0   , 'S' + 512},

  {0, 0, 0, 0}
};


//----------------------------------------------------------------------
// Function and class member implementation (function definitions)
//----------------------------------------------------------------------


cSDHOptions::cSDHOptions( char const* option_selection )
{
    std::string os( option_selection );

    if ( os.find( "general" ) !=  string::npos )
        usage.append( sdhusage_general );
    if ( os.find( "sdhcom_serial" ) != string::npos )
        usage.append( sdhusage_sdhcom_serial );
    if ( os.find( "sdhcom_common" ) != string::npos )
        usage.append( sdhusage_sdhcom_common );
    if ( os.find( "sdhcom_esdcan" ) != string::npos )
        usage.append( sdhusage_sdhcom_esdcan );
    if ( os.find( "sdhcom_peakcan" ) != string::npos )
        usage.append( sdhusage_sdhcom_peakcan );
    if ( os.find( "sdhcom_cancommon" ) != string::npos )
        usage.append( sdhusage_sdhcom_cancommon );
    if ( os.find( "sdhcom_tcp" ) != string::npos )
        usage.append( sdhusage_sdhcom_tcp );
    if ( os.find( "sdhother" ) != string::npos )
        usage.append( sdhusage_sdhother );
    if ( os.find( "dsacom" ) != string::npos )
        usage.append( sdhusage_dsacom );
    if ( os.find( "dsaother" ) != string::npos )
        usage.append( sdhusage_dsaother );
    if ( os.find( "dsaadjust" ) != string::npos )
        usage.append( sdhusage_dsaadjust );

    // set default options
    debug_level        = 0;    // 0: no debug messages
    debuglog           = &cerr;

    sdhport            = 0;    // 0=/dev/ttyS0=COM1
    strncpy( sdh_rs_device, "/dev/ttyS%d", MAX_DEV_LENGTH );
    timeout            = -1.0; // no timeout, wait forever (which is what we need)
    rs232_baudrate     = 115200;

    use_can_esd        = false;
    net                = 0;

    use_can_peak       = false;
    strncpy( sdh_canpeak_device, "/dev/pcanusb0", MAX_DEV_LENGTH );

    can_baudrate       = 1000000;
    id_read            = 43;
    id_write           = 42;

    use_tcp            = false;
    tcp_adr            = SDH_DEFAULT_TCP_ADR;
    tcp_port           = SDH_DEFAULT_TCP_PORT;

    use_radians        = false;
    use_fahrenheit     = false;
    period             = 0.0;  // no period, read only once

    dsaport            = 0;    // 0=/dev/ttyS0=COM1
    strncpy( dsa_rs_device, "/dev/ttyS%d", MAX_DEV_LENGTH );
    do_RLE             = true;

    framerate          = -1;
    fullframe          = false;
    sensorinfo         = false;
    controllerinfo     = false;

    for ( int i=0; i<6; i++ )
    {
        matrixinfo[i]  = -1;
        sensitivity[i] = -1.0;
        threshold[i]   = 65535;
    }
    reset_to_default   = false;
    persistent         = false;
    showdsasettings    = false;
}
//----------------------------------------------------------------------

cSDHOptions::~cSDHOptions()
{
    if ( debuglog && debuglog != &cerr )
        delete debuglog;
}
//----------------------------------------------------------------------

int cSDHOptions::Parse( int argc, char** argv,
                         char const* helptext, char const* progname, char const* version, char const* libname, char const* librelease )
{
    // parse options from command line
    unsigned long baudrate = 0;
    bool do_print_version = false;
    bool do_check_version = false;
    int option_index = 0;
    int rc;
    int i;
    int nb_matrixinfos = 0;

    while (1)
    {
        int c;
        c = getopt_long( argc, argv,
                         sdhoptions_short_options, sdhoptions_long_options,
                         &option_index );
        if (c == -1)
            break;

        switch (c)
        {
        case 'h':
            cout << helptext << "\n\nusage: " << progname << " [options]\n" << usage << "\n";
            exit(0);

        case 'v':
            do_print_version = true; // defer actual printing until all options are parsed (which might change the communication to use)
            break;

        case 'V':
            do_check_version = true; // defer actual checking until all options are parsed (which might change the communication to use)
            break;

        case 'd':
            rc = sscanf( optarg, "%d", &debug_level );
            assert( rc == 1 );
            break;

        case 'l':
        {
            ios_base::openmode mode = ios_base::out | ios_base::trunc;
            if ( optarg[0] == '+' )
            {
                mode = ios_base::app;
                optarg = optarg+1;
            }
            debuglog = new ofstream( optarg, mode );
            assert( debuglog != NULL );
            assert( ! debuglog->fail() );
            break;
        }

        case 'p':
            rc = sscanf( optarg, "%d", &sdhport );
            assert( rc == 1 );
            break;

        case 'S'+256:
            strncpy( sdh_rs_device, optarg, MAX_DEV_LENGTH );
            break;

        case 'T':
            rc = sscanf( optarg, "%lf", &timeout );
            assert( rc == 1 );
            break;

        case 'b':
            rc = sscanf( optarg, "%lu", &baudrate ); // store in intermediate variable since -b might be specified before --can
            assert( rc == 1 );
            break;

        //---
        case 'c':
            use_can_esd = true;
            break;

        case 'n':
            rc = sscanf( optarg, "%d", &net );
            assert( rc == 1 );
            break;

        //---
        case 'p'+256:
            use_can_peak = true;
            break;

        case 'P'+256:
            strncpy( sdh_canpeak_device, optarg, MAX_DEV_LENGTH );
            break;


        //---
        case 'e':
            if ( !strncmp( optarg, "0x", 2 ) || !strncmp( optarg, "0X", 2 ) )
                rc = sscanf( optarg, "0x%x", &id_read );
            else
                rc = sscanf( optarg, "%u", &id_read );
            assert( rc == 1 );
            break;

        case 'w':
            if ( !strncmp( optarg, "0x", 2 ) || !strncmp( optarg, "0X", 2 ) )
                rc = sscanf( optarg, "0x%x", &id_write );
            else
                rc = sscanf( optarg, "%u", &id_write );
            assert( rc == 1 );
            break;

        //---
        case 'R':
            use_radians = true;
            break;

        case 'F':
            use_fahrenheit = true;
            break;

        case 't':
            rc = sscanf( optarg, "%lf", &period );
            assert( rc == 1 );
            break;

        //---
        case 'q':
            rc = sscanf( optarg, "%d", &dsaport );
            assert( rc == 1 );
            break;

        case 'D'+256:
            strncpy( dsa_rs_device, optarg, MAX_DEV_LENGTH );
            break;

        case 256+'r':
            do_RLE = false;
            break;

        //---
        case 'r':
            rc = sscanf( optarg, "%d", &framerate );
            assert( rc == 1 );
            break;
        case 'f':
            fullframe = true;
            break;
        case 'S':
            sensorinfo = true;
            break;
        case 'C':
            controllerinfo = true;
            break;
        case 'M':
            if ( nb_matrixinfos > 5 )
            {
                cerr << "Error: matrixinfo requested more than 6 times!";
                exit( 1 );
            }
            rc = sscanf( optarg, "%d", &(matrixinfo[nb_matrixinfos++]) );
            assert( rc == 1 );
            break;

        //---
        case 999:
            rc = sscanf( optarg, "%lf", &(sensitivity[0]) ); // scan value
            assert( rc == 1 );
            assert( 0.0 <= sensitivity[0] );                 // check value
            assert( sensitivity[0] <= 1.0 );
            for ( i=1; i<6; i++)                             // copy value
                sensitivity[i] = sensitivity[0];
            break;
        case 1000:
        case 1001:
        case 1002:
        case 1003:
        case 1004:
        case 1005:
            i = c - 1000;
            rc = sscanf( optarg, "%lf", &(sensitivity[i]) );// scan value
            assert( rc == 1 );
            assert( 0.0 <= sensitivity[i] );               // check value
            assert( sensitivity[i] <= 1.0 );
            break;

        case 1009:
            rc = sscanf( optarg, "%d", &(threshold[0]) ); // scan value
            assert( rc == 1 );
            assert( 0 <= threshold[0] );                  // check value
            assert( threshold[0] <= 4095 );
            for ( i=1; i<6; i++)
                threshold[i] = threshold[0];              // copy value
            break;
        case 1010:
        case 1011:
        case 1012:
        case 1013:
        case 1014:
        case 1015:
            i = c - 1010;
            rc = sscanf( optarg, "%d", &(threshold[i]) ); // scan value
            assert( rc == 1 );
            assert( 0 <= threshold[i] );                  // check value
            assert( threshold[i] <= 4095 );
            break;

        case 'P'+512:
            persistent = true;
            break;

        case 'R'+512:
            reset_to_default = true;
            break;

        case 'S'+512:
            showdsasettings = true;
            break;

        case 'T'+256:
        {
            use_tcp = true;
            char* colon = strchr( optarg, (int) ':' );
            if ( colon )
            {
                if ( colon - optarg > 0 )
                {
                    *colon = '\0';
                    tcp_adr = optarg;
                }
                tcp_port = atoi( colon+1 );
            }
            else if ( strlen( optarg ) > 0 )
                tcp_adr = optarg;
            break;
        }
        //---

        default:
            cerr << "Error: getopt returned invalid character code '" << char(c) << "' = " << int(c) << ", giving up!\n";
            exit( 1 );
        }
    }

    if ( baudrate != 0 )
    {
        // a baudrate was specified on the command line, so overwrite the selected one
        if ( use_can_esd || use_can_peak )
            can_baudrate = baudrate;
        else
            rs232_baudrate = baudrate;
    }

    //-----
    if ( do_check_version )
    {
        try
        {
            g_sdh_debug_log = debuglog;
            cSDH hand( false, false, debug_level-1 );

            // Open configured communication to the SDH device
            OpenCommunication( hand );


            if ( hand.CheckFirmwareRelease() )
            {
                cout << "The firmware release of the connected SDH is the one recommended by\n";
                cout << "this SDHLibrary. Good.\n";
            }
            else
            {
                cout << "The firmware release of the connected SDH is NOT the one recommended\n";
                cout << "by this SDHLibrary:\n";
                cout << "  Actual SDH firmware release:      " << hand.GetFirmwareRelease() << "\n";
                cout << "  Recommended SDH firmware release: " << hand.GetFirmwareReleaseRecommended() << "\n";
                cout << "  => Communication with the SDH might not work as expected!\n";
            }
            hand.Close();
        }
        catch ( cSDHLibraryException* e )
        {
            cerr << "Could not check firmware release from SDH: " << e->what() << "\n";
            delete e;

            exit(1);
        }
    }
    //-----

    //-----
    char const* libdate = PROJECT_DATE;
    if ( do_print_version )
    {
        cout << "PC-side:\n";
        cout << "  Demo-program name:            " << argv[0] << "\n";
        cout << "  Demo-program revision:        " << version << "\n";
        cout << "  " << libname << " release:       " << librelease << "\n";
        cout << "  " << libname << " date:          " << libdate << "\n";
        cout << "  Recommended firmware release: " << cSDH::GetFirmwareReleaseRecommended() << "\n";
        cout.flush();

        try
        {
            g_sdh_debug_log = debuglog;
            cSDH hand( false, false, debug_level-1 );

            // Open configured communication to the SDH device
            OpenCommunication( hand );

            cout << "SDH-side:\n";
            cout << "  SDH firmware release:         " << hand.GetInfo( "release-firmware" ) << "\n";
            cout << "  SDH firmware date:            " << hand.GetInfo( "date-firmware" ) << "\n";
            cout << "  SDH SoC ID:                   " << hand.GetInfo( "release-soc" ) << "\n";
            cout << "  SDH SoC date:                 " << hand.GetInfo( "date-soc" ) << "\n";
            cout << "  SDH ID:                       " << hand.GetInfo( "id-sdh" ) << "\n";
            cout << "  SDH Serial Number:            " << hand.GetInfo( "sn-sdh" ) << "\n";
            hand.Close();
        }
        catch ( cSDHLibraryException* e )
        {
            cerr << "Could not get all version info from SDH: " << e->what() << "\n";
            delete e;
        }

        try
        {
            cDSA dsa( 0, dsaport );

            cout << "DSA-side:\n";
            cout << "  DSA controller info hw_version: " << (int) dsa.GetControllerInfo().hw_version << "\n";


            char buffer[8];
            UInt16 sw_version = dsa.GetControllerInfo().sw_version;
            snprintf( buffer, 8, "%hu", sw_version );
            std::string dsa_controller_sw_version( buffer );
            if ( sw_version >= 0x1000 )
            {
                // version numbering has changed, see bug 996
                std::string new_version_string( "" );
                std::string sep( "" );
                for ( int i=0; i < 4; i++ )
                {
                    snprintf( buffer, 8, "%hu", (sw_version >> (i*4)) & 0x0f );
                    new_version_string = buffer + sep + new_version_string;
                    sep = ".";
                }
                dsa_controller_sw_version += " (" + (new_version_string) + ")";
            }
            cout << "  DSA controller info sw_version: " << dsa_controller_sw_version << "\n";
            cout << "  DSA sensor info hw_revision:    " << (int) dsa.GetSensorInfo().hw_revision << "\n";
            dsa.Close();
        }
        catch ( cDSAException* e )
        {
            cerr << "Could not get sensor controller firmware release from DSA: " << e->what() << "\n";
            delete e;
        }

        exit( 0 );
    }
    //-----

    return optind;
    //
    //----------------------------------------------------------------------
}
//----------------------------------------------------------------------


void cSDHOptions::OpenCommunication( cSDH &hand )
{
    if      ( use_can_esd )
        // use CAN bus via an interface device from ESD
        hand.OpenCAN_ESD( net,                   // ESD CAN net number
                          can_baudrate,          // CAN baudrate in bit/s
                          timeout,               // timeout in s
                          id_read,               // CAN ID used for reading
                          id_write );            // CAN ID used for writing
    else if ( use_can_peak )
        // use CAN bus via an interface device from Peak
        hand.OpenCAN_PEAK( can_baudrate,         // CAN baudrate in bit/s
                           timeout,              // timeout in s
                           id_read,              // CAN ID used for reading
                           id_write,             // CAN ID used for writing
                           sdh_canpeak_device ); // PEAK CAN device to use (only on Linux)
    else if ( use_tcp )
        // use TCP/IP
        hand.OpenTCP( tcp_adr.c_str(),           // TCP address, IP or hostname
                      tcp_port,                  // TCP port number
                      timeout );                 // timeout in s
    else
        // use RS232
        hand.OpenRS232( sdhport,                 // RS232 port number
                        rs232_baudrate,          // baudrate
                        timeout,                 // timeout in s
                        sdh_rs_device );         // RS232 device (only on Linux)
}
//----------------------------------------------------------------------

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
