var exec = require('child_process').exec;
const remote = require('electron').remote;
var fs = require('fs');

module.exports = {
	analysis: function(datafile ,S) {
		var child;
		child = exec("python analysis.py " + datafile + " " + S , function (error, stdout, stderr) {
			if (stderr !== null ){
		    	console.log('stdout: ' + stdout);
				if (stderr !== null ){
					console.log('stderr: ' + stderr);
					return false
				}else if (error !== null ){
					console.log('exec error: ' + error);
					return false
				}else{
					return true
				};
			};
		});
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