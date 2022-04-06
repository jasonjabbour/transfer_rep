@echo off
rem  ---- Instructions ----
rem 1. start_full_pipeline.bat must be placed within the pipeline directory
rem 2. a shortcut for start_full_pipeline.bat must be placed in Win+R -> shell:startup
rem 3. must have user account control settings notifications off to start WampServer
rem -----------------------
start PCR_listen.bat
start PCR_voice_activity_detection.bat
start PCR_speaker_id.bat
start PCR_emotion_and_recommender.bat
start "" "..\cloud_upload\pcr_cloud\batch_files\cloud_start.bat"
start "" "C:\wamp\wampmanager.exe"