from obspy import UTCDateTime

def parseInputArgs( inputs ):
    
    input_args = dict()
    input_args['datasource'] = 
    input_args['nslc_list']  = []
    input_args['end']        = UTCDateTime.utcnow()
    input_args['start']      = input_args['end']-5*60 # 5 minutes ago
    
    
    # check input arguments. 1st argument is config file, 2nd is time (optional, otherwise right now)
    if len(inputs) == 1:		
        warnings.warn('Wrong input arguments. eg: main.py Pavlof_RSAM 201701020205')
        sys.exit()

    # no time given, use current time
    if len(input) == 2:
        # get current timestamp
        T0=UTCDateTime.utcnow()
        # round down to the nearest minute
        T0=UTCDateTime(T0.strftime('%Y-%m-%d %H:%M'))
    
    # time given, use it
    else:
        # time given as single string (eg. 201705130301)
        if len(sys.argv)==3:
            T0 = sys.argv[2]
        # time given as 2 strings (eg. 20170513 03:01)
        elif len(sys.argv)==4:
            T0='{}{}'.format(sys.argv[2],sys.argv[3])
        else:
            warnings.warn('Too many input arguments. eg: main.py Pavlof_RSAM 201701020205')
            sys.exit()		
        try:
            T0 = UTCDateTime(T0)
        except:
            warnings.warn('Needs end-time argument. eg: main.py Pavlof_RSAM 201701020205')
            sys.exit()
    try:
        # import the config file for the alarm you're running
        exec('import alarm_configs.{} as config'.format(sys.argv[1]))
        # import alarm module specified in config file
        ALARM=__import__('alarm_codes.'+config.alarm_type)

        # run the alarm
        eval('ALARM.{}.run_alarm(config,T0)'.format(config.alarm_type))

    # if error, send message to designated recipients
    except:
        print('Error...')
        b=traceback.format_exc()
        message = ''.join('{}\n'.format(a) for a in b.splitlines())
        message = '{}\n\n{}'.format(T0.strftime('%Y-%m-%d %H:%M'),message)
        subject=config.alarm_name+' error'
        attachment='alarm_aux_files/oops.jpg'
        utils.send_alert('Error',subject,message,attachment)
        
    return input_args