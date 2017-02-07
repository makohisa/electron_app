var exec = require('child_process').exec;
const remote = require('electron').remote;
var fs = require('fs');
const $ = require('jquery')

module.exports = {
	analysis: function(project_name,datafile) {
		var child;
		var deferred = new $.Deferred();
		child = exec("python analysis.py " + project_name + " " + datafile , function (error, stdout, stderr) {
			if (stderr !== null ){
		    	console.log('stdout: ' + stdout);
		    	if (stdout == "end") {
		    		deferred.resolve()
		    	}
				if (stderr !== null ){
					console.log('stderr: ' + stderr);
				}else if (error !== null ){
					console.log('exec error: ' + error);
				};
			};
		});
		return deferred.promise()
	},
	json_save: function(jsondata) {
		var json_path = "./data.json"

	  	fs.writeFile(json_path, jsondata, 'utf8', function (err) {
	    	if (err) {
	    		return console.log(err);
	    	}
	  	});
	},
	convertCSVtoObject: function(str){
	    var result = new Object()
	    var arr = new Array()
	    var tmp = str.split("\n"); 
	 
	    for(var i=0;i<tmp.length;++i){
	    	arr[i] = tmp[i].split(',');
	    	result[i] = new Object();

		    for(var j=0;j<arr[0].length;++j){
		    	result[i][arr[0][j]] = arr[i][j]
		    }
		}

	    return result
	},
	setup: function () {
		var child;
		child = exec(" which python", function (error, stdout, stderr) {
		  console.log('stdout: ' + stdout);
		  console.log('stderr: ' + stderr);
		  if (error !== null ){
		    console.log('exec error: ' + error);
		  };
		});
	}
}