CREATE SET VOLATILE TABLE all_mnd AS
	(SELECT 
		CAST(calendar_date AS DATE) /*(FORMAT'DDBMMMBYYYY') (CHAR(12)) */ AS month_interval_start
		--CONVERT(calendar_date, 
		,LAST_DAY(calendar_date) /* (FORMAT'DDBMMMBYYYY') (CHAR(12))*/  AS month_interval_end
		,Year(calendar_date) AS skadaar
		,Month(calendar_date) AS skadmnd
		,month_interval_end-month_interval_start+1 AS DaysInMonth
		FROM sys_calendar.CALENDAR
		WHERE (calendar_date BETWEEN '2010-01-01'  AND Now()) AND Extract (DAY FROM calendar_date)=1 ) 
	WITH DATA
	PRIMARY INDEX(month_interval_start)	ON COMMIT PRESERVE ROWS
	;
